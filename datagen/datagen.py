import random
import math
import networkx as nx
import hydra
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from samplers import class_from_cfg
from run_docker import *


def create_n_graph(nnodes):
    G = nx.Graph()
    for i in range(nnodes):
        G.add_node(i)
    return G


def create_graph_by_node_degrees(nnodes, node_degree_sampler):
    """ create graph according to degrees specified in the degree sampler
        (taken from challenge quickstart notebook code
    """
    G = create_n_graph(nnodes)
    node_degree = [node_degree_sampler() for _ in range(nnodes)]

    nodes = list(G.nodes)
    finish = False
    while True:
        aux_nodes = list(nodes)
        n0 = random.choice(aux_nodes)
        aux_nodes.remove(n0)
        # Remove adjacents nodes (only one link between two nodes)
        for n1 in G[n0]:
            if n1 in aux_nodes:
                aux_nodes.remove(n1)
        if len(aux_nodes) == 0:
            # No more links can be added to this node - can not accomplish node_degree for this node
            nodes.remove(n0)
            if len(nodes) == 1:
                break
            continue
        n1 = random.choice(aux_nodes)
        G.add_edge(n0, n1)

        for n in [n0, n1]:
            node_degree[n] -= 1
            if node_degree[n] == 0:
                nodes.remove(n)
                if len(nodes) == 1:
                    finish = True
                    break

        if finish:
            break

    return G


def generate_topology(net_size, cfg):
    if cfg.graph_creator.type == 'node_degree':
        node_degree_sampler = class_from_cfg(cfg.graph_creator.node_degree)

    node_scheduling_policy_sampler = class_from_cfg(cfg.node_scheduling_policy)
    node_buffer_size_sampler = class_from_cfg(cfg.node_buffer_size)
    link_bandwidth_sampler = class_from_cfg(cfg.link_bandwidth)
    wfq_sampler = class_from_cfg(cfg.wfq_weights)
    drr_sampler = class_from_cfg(cfg.drr_weights)
    if cfg.graph_creator.type == 'erdos_renyi':
        p_sampler = class_from_cfg(cfg.graph_creator.p)

    # create graph
    while True:
        if cfg.graph_creator.type == 'node_degree':
            G = create_graph_by_node_degrees(net_size, node_degree_sampler)
        elif cfg.graph_creator.type == 'erdos_renyi':
            G = nx.erdos_renyi_graph(net_size, p=p_sampler())
        else:
            raise RuntimeError('not implemented')

        if nx.is_connected(G):
            break

    # assign properties
    for i in range(net_size):
        # Assign to each node the scheduling Policy
        policy = node_scheduling_policy_sampler()
        G.nodes[i]["schedulingPolicy"] = policy
        if policy == 'WFQ':
            weights = wfq_sampler()
            assert(sum(weights) == 100)
            G.nodes[i]["schedulingWeights"] = ','.join([str(w) for w in weights])
        elif policy == 'DRR':
            weights = drr_sampler()
            assert(sum(weights) == 100)
            G.nodes[i]["schedulingWeights"] = ','.join([str(w) for w in weights])

        # Assign the buffer size of all the ports of the node
        G.nodes[i]["bufferSizes"] = node_buffer_size_sampler()

    for n0, n1 in G.edges:
        # Assign the link capacity to the link
        G[n0][n1]["bandwidth"] = link_bandwidth_sampler()

    return G


def write_graph(G, graph_file):
    nx.write_gml(G, graph_file)


def validate_graph(G):
    if G is None:
        return False

    if not nx.is_connected(G):
        return False

    for n, node in G.nodes().items():
        if node['schedulingPolicy'] not in ['FIFO', 'SP', 'WFQ', 'DRR']:
            return False
        if node['bufferSizes'] < 8000 or node['bufferSizes'] > 64000:
            return False

    for n0, n1 in G.edges:
        bw = G[n0][n1]["bandwidth"]
        if bw < 10000 or bw > 400000 or (bw % 1000) != 0:
            return False

    return True


def write_routing(G, routing, routing_file):
    with open(routing_file, "w") as r_fd:
        for src in G:
            for dst in G:
                if src == dst:
                    continue
                path = ','.join(str(x) for x in routing[src][dst])
                r_fd.write(path + "\n")


def generate_routing(G, cfg, routing_file=None):
    if cfg.type == 'shortest_path':
        paths = nx.shortest_path(G)
    else:
        sampler = class_from_cfg(cfg)
        if sampler is None:
            raise RuntimeError('not implemented')

        paths = sampler(G)

    if routing_file is not None:
        write_routing(G, paths, routing_file)

    return paths


def generate_tm(G, cfg, tm_file=None):
    bw_sampler = class_from_cfg(cfg.bandwidth)
    td_sampler = class_from_cfg(cfg.time_dist)
    on_sampler = class_from_cfg(cfg.td_on)
    off_sampler = class_from_cfg(cfg.td_off)
    sd_sampler = class_from_cfg(cfg.packet_dist)
    tos_sampler = class_from_cfg(cfg.tos)


    tm = {}
    for src in G:
        for dst in G:
            avg_bw = bw_sampler()
            assert(avg_bw >= 10 and avg_bw <= 10000)

            td = td_sampler()
            if td == 2:
                on = on_sampler()
                off = off_sampler()
            else:
                on, off = None, None
            sd = sd_sampler()
            assert (math.isclose(sum(sd['probs']), 1))
            sd_str = '0,' + ','.join(f'{sz},{p}' for sz, p in zip(sd['sizes'], sd['probs']))

            tos = tos_sampler()
            tm[(src, dst)] = {
                'avg_bw': avg_bw, 'td': td, 'td_on': on, 'td_off': off, 'sd': sd_str, 'tos': tos,
            }

    if tm_file is not None:
        write_tm(G, tm, tm_file)

    return tm


def assign_link_bw_from_traffic(G, routing, tm, cfg):
    bw_values = np.array(sorted(cfg.vals))

    links = defaultdict(list)
    for src in G:
        for dst in G:
            if src != dst:
                avg_bw = tm[(src, dst)]['avg_bw']
                path = routing[src][dst]
                for n1, n2 in zip(path[:-1], path[1:]):
                    links[(n1,n2)].append(avg_bw)

    for (n1,n2), vals in links.items():
        demand = sum(vals)
        if cfg.assignment == 'exact':
            bw = demand // 1000 * 1000
        elif cfg.assignment == 'nearest':
            i = bw_values.searchsorted(demand)
            if i == len(bw_values):
                i = len(bw_values)-1
            elif i > 0 and (demand-bw_values[i-1])<(bw_values[i]-demand):
                i -= 1
            bw = bw_values[i].item()
        elif cfg.assignment == 'above':
            i = bw_values.searchsorted(demand)
            if i == len(bw_values):
                i = len(bw_values)-1
            bw = bw_values[i].item()
        elif cfg.assignment == 'below':
            i = bw_values.searchsorted(demand)-1
            if i < 0:
                i = 0
            bw = bw_values[i].item()

        G[n1][n2]["bandwidth"] = bw


def validate_sd(sd):
    vv = sd.split(',')[1:]

    sizes = [int(vv[i]) for i in range(0, len(vv), 2)]
    for sz in sizes:
        if sz < 256 or sz > 2000:
            raise RuntimeError('validate_sd failed: bad pkt size')

    probs = [float(vv[i]) for i in range(1, len(vv), 2)]
    for p in probs:
        if p < 0 or p > 1:
            raise RuntimeError('validate_sd failed: bad pkt prob')

    total_p = sum(probs)
    if not math.isclose(total_p, 1):
        raise RuntimeError('validate_sd failed: total prob not 1')

    return True


def validate_tm(G, tm):
    for src in G:
        for dst in G:
            d = tm[(src, dst)]
            if d['avg_bw'] < 10 or d['avg_bw'] > 10000:
                raise RuntimeError('validate_tm failed: bad avg bw')
            if d['td'] not in [0, 1, 2]:
                raise RuntimeError('validate_tm failed: bad td')
            if not validate_sd(d['sd']):
                raise RuntimeError('validate_tm failed: bad sd')
            if not d['tos'] in [0, 1, 2]:
                return RuntimeError('validate_tm failed: bad tos')

    return True


def write_tm(G, tm, tm_file):
    with open(tm_file, "w") as tm_fd:
        for src in G:
            for dst in G:
                d = tm[(src, dst)]
                td = d['td']
                if td == 2:
                    td = '{},{},{}'.format(td, d['td_on'], d['td_off'])
                else:
                    td = str(td)

                traffic_line = "{},{},{},{},{},{}".format(
                    src, dst, d['avg_bw'], td, d['sd'], d['tos'])
                tm_fd.write(traffic_line + "\n")


def main(cfg):
    random.seed()
    np.random.seed()

    save_path = Path(cfg.save_path)
    if save_path.exists():
        print('WARNING: save path already exists: ', save_path)
    else:
        save_path.mkdir(parents=True)

    graphs_path = save_path / 'graphs'
    graphs_path.mkdir(exist_ok=False)

    routings_path = save_path / 'routings'
    routings_path.mkdir(exist_ok=False)

    tm_path = save_path / 'tm'
    tm_path.mkdir(exist_ok=False)

    simulation_file = save_path / 'simulation.txt'
    if simulation_file.exists():
        raise RuntimeError('simulation file already exists')

    size_sampler = class_from_cfg(cfg.topology.net_size)
    for i_topo in tqdm(range(cfg.num_topologies), desc='topology'):
        net_size = size_sampler()
        topo_name = f'{i_topo:05d}'
        G = None
        while not validate_graph(G):
            G = generate_topology(net_size, cfg.topology)

        graph_file = graphs_path / f'graph_{topo_name}.txt'

        routing = generate_routing(G, cfg.routing)
        routing_file = routings_path / f'routing_{topo_name}.txt'
        # print(routing)

        for i_tm in tqdm(range(cfg.num_tm_per_topology), desc='traffic'):
            tm = generate_tm(G, cfg.traffic)
            assert(validate_tm(G, tm))
            # print(tm)
            tm_file = tm_path / f'tm_{topo_name}_{i_tm:05d}.txt'
            write_tm(G, tm, tm_file)

            with open(simulation_file, 'a') as sim_fd:
                sim_line = "{},{},{}\n".format(graph_file, routing_file, tm_file)
                # If dataset has been generated in windows, convert paths into linux format
                sim_fd.write(sim_line.replace("\\", "/"))

        if cfg.topology.link_bandwidth.type == 'LinkBWFromTraffic':
            assign_link_bw_from_traffic(G, routing, tm, cfg.topology.link_bandwidth)

        write_graph(G, graph_file)
        write_routing(G, routing, routing_file)

    # docker prep
    save_docker_config(save_path, cfg.docker)
    docker_cmd(save_path)


@hydra.main(config_path="../config/datagen")
def hydra_main(cfg):
    # this allows to add new fields to cfg programmatically
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    main(cfg)


if __name__ == "__main__":
    hydra_main()
