config = {
### Fixed.
'batch_size': 64,
'use_vanilla_rnn': True,
'teacher_use_vanilla_rnn': True,
'save_weights': True,
'save_logs': True,

### Options.
'lr': 0.001,
'n_iter': 50,

# Student network.
'rnn_arch': '6',
'rec_sparsity': 0.2,

# Teacher network.
'teacher_rnn_arch': '6',
'teacher_rec_sparsity': 0.2,
}