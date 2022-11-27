#!/usr/bin/env bash

# uncomment if using conda environment and replace "gnnch" with your env name
#eval "$(conda shell.bash hook)"
#conda activate gnnch

# code root directory (containing datagen/datagen.py)
src=/home/yakovl/dev/GNNetworkingChallenge

# directory where to create the datasets (must exist)
datagen_root=$src/generated_datasets

# dataset 0
python $src/datagen/datagen.py --config-name config topology.net_size=6,7,8,9,10 num_topologies=20 num_tm_per_topology=100 ds_name='random_${num_topologies}x${num_tm_per_topology}_netsz_${topology.net_size}' datasets_root=${datagen_root}/0 -m

# dataset 1
python $src/datagen/datagen.py --config-name config topology.net_size=10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10 num_topologies=25 num_tm_per_topology=20 ds_name='netsz_${topology.net_size}_${num_topologies}x${num_tm_per_topology}' datasets_root=${datagen_root}/1 -m

# dataset 2
python $src/datagen/datagen.py --config-name config topology.net_size=7,7,7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8,8,8 num_topologies=25 num_tm_per_topology=20 ds_name='netsz_${topology.net_size}_${num_topologies}x${num_tm_per_topology}' datasets_root=${datagen_root}/2 -m

# dataset 3
python $src/datagen/datagen.py --config-name config topology.net_size=10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9 num_topologies=10 num_tm_per_topology=10 ds_name='netsz_${topology.net_size}_${num_topologies}x${num_tm_per_topology}_p${topology.graph_creator.p}' datasets_root=${datagen_root}/3 topology.graph_creator.p=0.3,0.5 -m

# dataset 4
python $src/datagen/datagen.py --config-name config topology.net_size=10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7 num_topologies=10 num_tm_per_topology=10 ds_name='netsz_${topology.net_size}_${num_topologies}x${num_tm_per_topology}_p${topology.graph_creator.p}' datasets_root=${datagen_root}/4 topology.graph_creator.p=0.3,0.5 -m

# dataset 5
python $src/datagen/datagen.py --config-name config topology.net_size=10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8 num_topologies=35 num_tm_per_topology=10 ds_name='netsz_${topology.net_size}_${num_topologies}x${num_tm_per_topology}_p01-04' +topology/graph_creator=erdos_renyi_p01-04 datasets_root=${datagen_root}/5 -m

# dataset 6
python $src/datagen/datagen.py --config-name config_asval topology.net_size=10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8 num_topologies=35 num_tm_per_topology=10 ds_name='byval_netsz_${topology.net_size}_${num_topologies}x${num_tm_per_topology}_p01-04' datasets_root=${datagen_root}/6 -m

# dataset 7
python $src/datagen/datagen.py --config-name config_asval_mixroute topology.net_size=10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8 num_topologies=35 num_tm_per_topology=10 ds_name='byval_route_by_linkbw_netsz_${topology.net_size}_${num_topologies}x${num_tm_per_topology}_p01-04' +routing=shortest_by_link_bw datasets_root=${datagen_root}/7 -m

# dataset 8
python $src/datagen/datagen.py --config-name config_asval_mixroute topology.net_size=10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8 num_topologies=35 num_tm_per_topology=10 ds_name='byval_route_random_netsz_${topology.net_size}_${num_topologies}x${num_tm_per_topology}_p01-04' +routing=random_path datasets_root=${datagen_root}/8 -m

# dataset 9
python $src/datagen/datagen.py --config-name config_asval_mixroute topology.net_size=10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8 num_topologies=35 num_tm_per_topology=10 ds_name='byval_route_by_linkbw_netsz_${topology.net_size}_${num_topologies}x${num_tm_per_topology}_p01-04' +routing=shortest_by_link_bw datasets_root=${datagen_root}/9 -m

# dataset 10
python $src/datagen/datagen.py --config-name config_asval_mixroute topology.net_size=10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8 num_topologies=35 num_tm_per_topology=10 ds_name='byval_route_random_netsz_${topology.net_size}_${num_topologies}x${num_tm_per_topology}_p01-04' +routing=random_path datasets_root=${datagen_root}/10 -m

# dataset 11
python $src/datagen/datagen.py --config-name config_asval_rndpath topology.net_size=10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10 num_topologies=35 num_tm_per_topology=10 ds_name='byval_rndpath_bylength_netsz_${topology.net_size}_${num_topologies}x${num_tm_per_topology}_p01-04' datasets_root=${datagen_root}/11 -m

# dataset 12
python $src/datagen/datagen.py --config-name config_asval_rndpath2 topology.net_size=10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10 num_topologies=35 num_tm_per_topology=10 ds_name='byval_rndpath2_bylength_netsz_${topology.net_size}_${num_topologies}x${num_tm_per_topology}_p01-04' datasets_root=${datagen_root}/12 -m

# dataset 13
python $src/datagen/datagen.py --config-name config_asval_rndpath2 topology.net_size=10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10 num_topologies=35 num_tm_per_topology=10 ds_name='byval_rndpath2_bylength_netsz_${topology.net_size}_${num_topologies}x${num_tm_per_topology}_p01-04' datasets_root=${datagen_root}/13 -m

# dataset 14
python $src/datagen/datagen.py --config-name config_asval_rndpath2 topology.net_size=8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9 num_topologies=35 num_tm_per_topology=10 ds_name='byval_rndpath2_bylength_netsz_${topology.net_size}_${num_topologies}x${num_tm_per_topology}_p01-04' datasets_root=${datagen_root}/14 -m

# dataset 15
python $src/datagen/datagen.py --config-name config_asval_rndpath3 topology.net_size=8,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,10,10 num_topologies=35 num_tm_per_topology=10 ds_name='config_asval_rndpath3_netsz_${topology.net_size}_${num_topologies}x${num_tm_per_topology}_p01-04' datasets_root=${datagen_root}/15 -m

# dataset hard1
for lbw in 10k-25k_1 ; do
	for bsz in uniform small_1 ; do
		python $src/datagen/datagen.py --config-name hard1 topology.net_size=8,9,9,10,10 num_topologies=10 num_tm_per_topology=10 topology/link_bandwidth=$lbw topology/node_buffer_size=$bsz topology.graph_creator.p=0.3,0.4,0.5,0.6 ds_name='hard1_lbw_'$lbw'_bsz_'$bsz'_netsz_${topology.net_size}_${num_topologies}x${num_tm_per_topology}_p_${topology.graph_creator.p}' datasets_root=${datagen_root}/hard1 -m
  done
done

# dataset hard2
for lbw in 10k-25k_1 10k-25k_2 10k-25k_3 10k-25k_4 10k-25k_5 25k-40k ; do
	for bsz in uniform small_1 ; do
		python $src/datagen/datagen.py --config-name hard1 topology.net_size=8,9,9,10,10 num_topologies=10 num_tm_per_topology=10 topology/link_bandwidth=$lbw topology/node_buffer_size=$bsz topology.graph_creator.p=0.3,0.4,0.5,0.6 ds_name='hard1_lbw_'$lbw'_bsz_'$bsz'_netsz_${topology.net_size}_${num_topologies}x${num_tm_per_topology}_p_${topology.graph_creator.p}' datasets_root=${datagen_root}/hard2 -m
  done
done


# dataset hard3
p=0.3,0.4,0.5
net_sizes=9,9,10,10,10

for lbw in from_traffic ; do
	for bsz in small_2 ; do
		for avgbw in large_1 large_2 asval ; do
			python $src/datagen/datagen.py --config-name hard3 topology.net_size=$net_sizes num_topologies=100 num_tm_per_topology=1 topology/link_bandwidth=$lbw topology/node_buffer_size=$bsz traffic/bandwidth=$avgbw topology.link_bandwidth.assignment=nearest,above topology.graph_creator.p=$p ds_name='hard3_lbw_'$lbw'-${topology.link_bandwidth.assignment}_bsz_'$bsz'_avgbw_'$avgbw'_netsz_${topology.net_size}_${num_topologies}x${num_tm_per_topology}_p_${topology.graph_creator.p}' datasets_root=${datagen_root}/hard3 -m
		done
  done
done

for lbw in 10k-25k_1 25k-40k 25k-40k_like8 ; do
	for bsz in small_2 ; do
		for avgbw in large_1 large_2 ; do
			python $src/datagen/datagen.py --config-name hard3 topology.net_size=$net_sizes num_topologies=10 num_tm_per_topology=10 topology/link_bandwidth=$lbw topology/node_buffer_size=$bsz traffic/bandwidth=$avgbw topology.graph_creator.p=$p ds_name='hard3_lbw_'$lbw'_bsz_'$bsz'_avgbw_'$avgbw'_netsz_${topology.net_size}_${num_topologies}x${num_tm_per_topology}_p_${topology.graph_creator.p}' datasets_root=${datagen_root}/hard3 -m
		done
  done
done

for lbw in 25k-40k_like8 ; do
	for bsz in small_2 ; do
		for avgbw in asval ; do
			python $src/datagen/datagen.py --config-name hard3 topology.net_size=$net_sizes num_topologies=10 num_tm_per_topology=10 topology/link_bandwidth=$lbw topology/node_buffer_size=$bsz traffic/bandwidth=$avgbw topology.graph_creator.p=$p ds_name='hard3_lbw_'$lbw'_bsz_'$bsz'_avgbw_'$avgbw'_netsz_${topology.net_size}_${num_topologies}x${num_tm_per_topology}_p_${topology.graph_creator.p}' datasets_root=${datagen_root}/hard3 -m
		done
  done
done


# dataset hard4
p=0.3,0.4,0.5
net_sizes=9,9,10,10,10

for lbw in 25k-40k_like8_easier1 25k-40k_like8 ; do
	for bsz in uniform small_1 ; do
		for avgbw in asval large_1 large_2 ; do
			python $src/datagen/datagen.py --config-name hard3 topology.net_size=$net_sizes num_topologies=10 num_tm_per_topology=10 topology/link_bandwidth=$lbw topology/node_buffer_size=$bsz traffic/bandwidth=$avgbw topology.graph_creator.p=$p ds_name='hard3_lbw_'$lbw'_bsz_'$bsz'_avgbw_'$avgbw'_netsz_${topology.net_size}_${num_topologies}x${num_tm_per_topology}_p_${topology.graph_creator.p}' datasets_root=${datagen_root}/hard4 -m
		done
  done
done


# dataset hard5
p=0.3,0.4
net_sizes=10,10,9,9,8
for lbw in from_traffic ; do
  for bsz in small_1 small_2 ; do
		for avgbw in uniform2k uniform3k large_2 ; do
		  for pkt in small1 ; do
        if [ "$lbw" = "from_traffic" ]; then
          echo "from traffic"
          python $src/datagen/datagen.py --config-name hard5 topology.net_size=$net_sizes num_topologies=130 num_tm_per_topology=1 topology/link_bandwidth=$lbw topology/node_buffer_size=$bsz traffic/bandwidth=$avgbw topology.link_bandwidth.assignment=above topology.graph_creator.p=$p traffic/packet_dist=$pkt ds_name='hard5_lbw_'$lbw'-${topology.link_bandwidth.assignment}_bsz_'$bsz'_avgbw_'$avgbw'_pkt_'$pkt'_netsz_${topology.net_size}_${num_topologies}x${num_tm_per_topology}_p_${topology.graph_creator.p}' datasets_root=${datagen_root}/hard5 -m
        else
          echo "Slbw"
          python $src/datagen/datagen.py --config-name hard5 topology.net_size=$net_sizes num_topologies=130 num_tm_per_topology=1 topology/link_bandwidth=$lbw topology/node_buffer_size=$bsz traffic/bandwidth=$avgbw topology.graph_creator.p=$p traffic/packet_dist=$pkt ds_name='hard5_lbw_'$lbw'_bsz_'$bsz'_avgbw_'$avgbw'_pkt_'$pkt'_netsz_${topology.net_size}_${num_topologies}x${num_tm_per_topology}_p_${topology.graph_creator.p}' datasets_root=${datagen_root}/hard5 -m
        fi
			done
		done
  done
done


# dataset hard6
p=0.4,0.3,0.4,0.3,0.4,0.3,0.4
net_sizes=10,10,9,9,8,10,10,9,9,8
for lbw in from_traffic ; do
  for avgbw in uniform2k-6k ; do
    for bsz in small_1; do
		  for pkt in small1 ; do
        if [ "$lbw" = "from_traffic" ]; then
          echo "from traffic"
          python $src/datagen/datagen.py --config-name hard5 topology.net_size=$net_sizes num_topologies=110 num_tm_per_topology=1 topology/link_bandwidth=$lbw topology/node_buffer_size=$bsz traffic/bandwidth=$avgbw topology.link_bandwidth.assignment=above topology.graph_creator.p=$p traffic/packet_dist=$pkt ds_name='hard5_lbw_'$lbw'-${topology.link_bandwidth.assignment}_bsz_'$bsz'_avgbw_'$avgbw'_pkt_'$pkt'_netsz_${topology.net_size}_${num_topologies}x${num_tm_per_topology}_p_${topology.graph_creator.p}' datasets_root=${datagen_root}/hard6 -m
        else
          echo "Slbw"
          python $src/datagen/datagen.py --config-name hard5 topology.net_size=$net_sizes num_topologies=110 num_tm_per_topology=1 topology/link_bandwidth=$lbw topology/node_buffer_size=$bsz traffic/bandwidth=$avgbw topology.graph_creator.p=$p traffic/packet_dist=$pkt ds_name='hard5_lbw_'$lbw'_bsz_'$bsz'_avgbw_'$avgbw'_pkt_'$pkt'_netsz_${topology.net_size}_${num_topologies}x${num_tm_per_topology}_p_${topology.graph_creator.p}' datasets_root=${datagen_root}/hard6 -m
        fi
			done
		done
  done
done


# dataset hard5_small6
tgtdir=${datagen_root}/hard5_small6
p=0.4
net_sizes=10,10,10,10,10,10,10,10,10,10
for lbw in from_traffic 100k_3 100k_4 ; do
  for bsz in small_1 small_2 ; do
		for avgbw in uniform4.5k uniform2k-6k ; do
		  for pkt in small1 ; do
        if [ "$lbw" = "from_traffic" ]; then
          echo "from traffic"
          python $src/datagen/datagen.py --config-name hard5 topology.net_size=$net_sizes num_topologies=1 num_tm_per_topology=1 topology/link_bandwidth=$lbw topology/node_buffer_size=$bsz traffic/bandwidth=$avgbw topology.link_bandwidth.assignment=above topology.graph_creator.p=$p traffic/packet_dist=$pkt ds_name='hard5_lbw_'$lbw'-${topology.link_bandwidth.assignment}_bsz_'$bsz'_avgbw_'$avgbw'_pkt_'$pkt'_netsz_${topology.net_size}_${num_topologies}x${num_tm_per_topology}_p_${topology.graph_creator.p}' datasets_root=$tgtdir -m
        else
          echo "Slbw"
          python $src/datagen/datagen.py --config-name hard5 topology.net_size=$net_sizes num_topologies=1 num_tm_per_topology=1 topology/link_bandwidth=$lbw topology/node_buffer_size=$bsz traffic/bandwidth=$avgbw topology.graph_creator.p=$p traffic/packet_dist=$pkt ds_name='hard5_lbw_'$lbw'_bsz_'$bsz'_avgbw_'$avgbw'_pkt_'$pkt'_netsz_${topology.net_size}_${num_topologies}x${num_tm_per_topology}_p_${topology.graph_creator.p}' datasets_root=$tgtdir -m
        fi
			done
		done
  done
done


# dataset hard5_small5
tgtdir=${datagen_root}/hard5_small5
p=0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3
net_sizes=10
for lbw in from_traffic ; do
  for bsz in small_1 ; do
		for avgbw in uniform2k uniform3k uniform4.5k ; do
		  for pkt in small1 ; do
        python $src/datagen/datagen.py --config-name hard5 topology.net_size=$net_sizes num_topologies=1 num_tm_per_topology=1 topology/link_bandwidth=$lbw topology/node_buffer_size=$bsz traffic/bandwidth=$avgbw topology.link_bandwidth.assignment=above topology.graph_creator.p=$p traffic/packet_dist=$pkt ds_name='hard5_lbw_'$lbw'-${topology.link_bandwidth.assignment}_bsz_'$bsz'_avgbw_'$avgbw'_pkt_'$pkt_'_bsz_'$bsz'_avgbw_'$avgbw'_netsz_${topology.net_size}_${num_topologies}x${num_tm_per_topology}_p_${topology.graph_creator.p}' datasets_root=$tgtdir -m
			done
		done
  done
done


# dataset hard5_small4
tgtdir=${datagen_root}/hard5_small4
p=0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3
net_sizes=10
for lbw in from_traffic ; do
  for bsz in small_1 ; do
		for avgbw in large_2 ; do
		  for pkt in small1 ; do
        python $src/datagen/datagen.py --config-name hard5 topology.net_size=$net_sizes num_topologies=4 num_tm_per_topology=1 topology/link_bandwidth=$lbw topology/node_buffer_size=$bsz traffic/bandwidth=$avgbw topology.link_bandwidth.assignment=above topology.graph_creator.p=$p traffic/packet_dist=$pkt ds_name='hard5_lbw_'$lbw'-${topology.link_bandwidth.assignment}_bsz_'$bsz'_avgbw_'$avgbw'_pkt_'$pkt_'_bsz_'$bsz'_avgbw_'$avgbw'_netsz_${topology.net_size}_${num_topologies}x${num_tm_per_topology}_p_${topology.graph_creator.p}' datasets_root=$tgtdir -m
			done
		done
  done
done


# dataset hard5_small3
tgtdir=${datagen_root}/hard5_small3
p=0.3
net_sizes=10
for lbw in 100k_1 100k_2 100k_3 ; do
	for bsz in small_1 small_2 ; do
		for avgbw in uniform2k uniform3k uniform4.5k ; do
			python $src/datagen/datagen.py --config-name hard3 routing.path_sampler=longest_path topology.net_size=$net_sizes num_topologies=10 num_tm_per_topology=1 topology/link_bandwidth=$lbw topology/node_buffer_size=$bsz traffic/bandwidth=$avgbw topology.graph_creator.p=$p ds_name='hard3_lbw_'$lbw'_bsz_'$bsz'_avgbw_'$avgbw'_netsz_${topology.net_size}_${num_topologies}x${num_tm_per_topology}_p_${topology.graph_creator.p}_longpath' datasets_root=$tgtdir -m
		done
  done
done

# dataset hard5_small2
tgtdir=${datagen_root}/hard5_small2
p=0.3
net_sizes=9,10
for lbw in 100k_1 100k_2 100k_3 ; do
	for bsz in small_1 small_2 ; do
		for avgbw in uniform2k uniform3k uniform4.5k ; do
			python $src/datagen/datagen.py --config-name hard3 topology.net_size=$net_sizes num_topologies=10 num_tm_per_topology=1 topology/link_bandwidth=$lbw topology/node_buffer_size=$bsz traffic/bandwidth=$avgbw topology.graph_creator.p=$p ds_name='hard3_lbw_'$lbw'_bsz_'$bsz'_avgbw_'$avgbw'_netsz_${topology.net_size}_${num_topologies}x${num_tm_per_topology}_p_${topology.graph_creator.p}' datasets_root=$tgtdir -m
		done
  done
done

# dataset hard5_small
tgtdir=${datagen_root}/hard5_small
p=0.3
net_sizes=9,10
for lbw in 40k-100k_1 40k-100k_2 40k-100k_3 ; do
	for bsz in small_1 small_2 ; do
		for avgbw in large_1 large_2 ; do
			python $src/datagen/datagen.py --config-name hard3 topology.net_size=$net_sizes num_topologies=10 num_tm_per_topology=1 topology/link_bandwidth=$lbw topology/node_buffer_size=$bsz traffic/bandwidth=$avgbw topology.graph_creator.p=$p ds_name='hard3_lbw_'$lbw'_bsz_'$bsz'_avgbw_'$avgbw'_netsz_${topology.net_size}_${num_topologies}x${num_tm_per_topology}_p_${topology.graph_creator.p}' datasets_root=$tgtdir -m
		done
  done
done

