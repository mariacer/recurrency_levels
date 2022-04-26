config = {
'lr': 0.001,
'n_iter': 200,
'batch_size': 64,
'data_random_seed': 64,

# Student network.
'use_vanilla_rnn': True,
'rnn_arch': '32',
'rec_sparsity': 0.2,

# Teacher network.
'teacher_use_vanilla_rnn': True,
'teacher_rnn_arch': '32',
'teacher_rec_sparsity': 0.4,
'teacher_n_ts_in': 500,

'compute_late_mse': True,
'save_weights': True,
'save_logs': True,
}
