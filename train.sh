# Note: Executing this script to re-compute the full data set requires approximately 17500 core-hours (~2 years) of compute
#       -- assuming a Intel Xeon E3-1240 v6 4-Core CPU with a clock-speed of 3.7 GHz.
#       Consider instead downloading pre-computed data as described in README.

#########################
#  Main results on DQN  #
#########################  ~2500 core-hours
for RUN in {0..99}
do
  python train.py --env_load a3-main --model classical --width_nn 16 --experiment "$RUN" --overwrite  # ~0.5 core-hours
  python train.py --env_load a3-main --model classical --width_nn 32 --experiment "$RUN" --overwrite  # ~0.5 core-hours
  python train.py --env_load a3-main --model classical --width_nn 64 --experiment "$RUN" --overwrite  # ~0.5 core-hours
  python train.py --env_load a3-main --model quantum --qubits_qnn 10 --experiment "$RUN" --overwrite  # ~2.7 core-hours
  python train.py --env_load a3-main --model quantum --qubits_qnn 14 --experiment "$RUN" --overwrite  # ~20.8 core-hours
done

###################################
# Comparison of Classical Models  #
###################################  ~400 core-hours
for RUN in {0..99}
do
  python train.py --env_load a3-main --model classical --width_nn 32 --depth_nn 1 --experiment "$RUN" --overwrite  # ~0.5 core-hours
  python train.py --env_load a3-main --model classical --width_nn 32 --depth_nn 3 --experiment "$RUN" --overwrite  # ~0.5 core-hours
  python train.py --env_load a3-main --model classical --width_nn 32 --depth_nn 4 --experiment "$RUN" --overwrite  # ~0.5 core-hours
  python train.py --env_load a3-main --model classical --width_nn 64 --depth_nn 1 --experiment "$RUN" --overwrite  # ~0.5 core-hours
  python train.py --env_load a3-main --model classical --width_nn 64 --depth_nn 3 --experiment "$RUN" --overwrite  # ~0.5 core-hours
  python train.py --env_load a3-main --model classical --width_nn 64 --depth_nn 4 --experiment "$RUN" --overwrite  # ~0.5 core-hours
  python train.py --env_load a3-main --model classical --width_nn 128 --depth_nn 2 --experiment "$RUN" --overwrite  # ~0.5 core-hours
  python train.py --env_load a3-main --model classical --width_nn 256 --depth_nn 2 --experiment "$RUN" --overwrite  # ~0.5 core-hours
done

##################################
#  Comparison of Quantum Models  #
##################################  ~2560 core-hours
for RUN in {0..99}
do
  python train.py --env_load a3-main --model quantum --qubits_qnn 6 --layers_qnn 4 --experiment "$RUN" --overwrite  # ~1.2 core-hours
  python train.py --env_load a3-main --model quantum --qubits_qnn 8 --layers_qnn 3 --experiment "$RUN" --overwrite  # ~1.3 core-hours
  python train.py --env_load a3-main --model quantum --qubits_qnn 8 --layers_qnn 4 --experiment "$RUN" --overwrite  # ~1.6 core-hours
  python train.py --env_load a3-main --model quantum --qubits_qnn 8 --layers_qnn 5 --experiment "$RUN" --overwrite  # ~1.9 core-hours
  python train.py --env_load a3-main --model quantum --qubits_qnn 8 --layers_qnn 6 --experiment "$RUN" --overwrite  # ~3.3 core-hours
  python train.py --env_load a3-main --model quantum --qubits_qnn 10 --layers_qnn 3 --experiment "$RUN" --overwrite  # ~2.2 core-hours
  python train.py --env_load a3-main --model quantum --qubits_qnn 10 --layers_qnn 5 --experiment "$RUN" --overwrite  # ~3.3 core-hours
  python train.py --env_load a3-main --model quantum --qubits_qnn 10 --layers_qnn 6 --experiment "$RUN" --overwrite  # ~3.8 core-hours
  python train.py --env_load a3-main --model quantum --qubits_qnn 12 --layers_qnn 4 --experiment "$RUN" --overwrite  # ~7.0 core-hours
done

##################################
#  Increasing Trajectory Degree  #
##################################  ~1260 core-hours
for DEGREE in 4 5 6
do
  for RUN in {0..99}
  do
    python train.py --env_load a3-main --env_degree "$DEGREE" --model classical --width_nn 16 --experiment "$RUN" --overwrite  # ~0.5 core-hours
    python train.py --env_load a3-main --env_degree "$DEGREE" --model classical --width_nn 32 --experiment "$RUN" --overwrite  # ~0.5 core-hours
    python train.py --env_load a3-main --env_degree "$DEGREE" --model classical --width_nn 64 --experiment "$RUN" --overwrite  # ~0.5 core-hours
    python train.py --env_load a3-main --env_degree "$DEGREE" --model quantum --qubits_qnn 10 --experiment "$RUN" --overwrite  # ~2.7 core-hours
  done
done

#####################################
#  Multiple Antenna Configurations  #
#####################################  ~8400 core-hours
for ANTENNAS in 2 3 4 5
do
  for ID in {0..4}
  do
    python train.py --env_load a"$ANTENNAS"-"$ID" --model classical --width_nn 16 --experiment "$RUN" --overwrite  # ~0.5 core-hours
    python train.py --env_load a"$ANTENNAS"-"$ID" --model classical --width_nn 32 --experiment "$RUN" --overwrite  # ~0.5 core-hours
    python train.py --env_load a"$ANTENNAS"-"$ID" --model classical --width_nn 64 --experiment "$RUN" --overwrite  # ~0.5 core-hours
    python train.py --env_load a"$ANTENNAS"-"$ID" --model quantum --qubits_qnn 10 --experiment "$RUN" --overwrite  # ~2.7 core-hours
  done
done

#################################
#  Training with PPO Algorithm  #
#################################  ~2100 core-hours
for RUN in {0..99}
do
  python train.py --env_load a3-main --method ppo --epochs_train 500 --model classical --width_nn 16 --epsilon_clip 0.1 --learning_rate 0.001 --experiment "$RUN" --overwrite  # ~2.5 core-hours
  python train.py --env_load a3-main --method ppo --epochs_train 500 --model classical --width_nn 32 --epsilon_clip 0.1 --learning_rate 0.001 --experiment "$RUN" --overwrite  # ~2.5 core-hours
  python train.py --env_load a3-main --method ppo --epochs_train 500 --model classical --width_nn 64 --epsilon_clip 0.1 --learning_rate 0.001 --experiment "$RUN" --overwrite  # ~2.5 core-hours
  python train.py --env_load a3-main --method ppo --epochs_train 500 --model quantum --qubits_qnn 10 --epsilon_clip 0.2 --learning_rate 0.001 --learning_rate_quantum 0.001 --experiment "$RUN" --overwrite  # ~13.5 core-hours
done
