# Recurrency levels in RNNs

In this project, we study the potential relationship between levels of recurrency in an RNN and their performance in specific tasks.
The code is based on a previous implementation by Christian Henning, Maria Cervera and Benjamin Ehret for another <a href="hhttps://github.com/mariacer/cl_in_rnns">project</a>.

## Experiments

For running experiments on real-world sequential tasks, move to the [real_world_benchmarks](real_world_benchmarks) folder and exectue from there:

- ``run_pos.py``: For a Part-of-Speech tagging experiment
- ``run_audioset.py``: To run classification of audio samples from the Audioset dataset

For running experiments on a student-teacher setting, move to the [student_teacher](student_teacher) folder and execute from there:

- ``run_student_teacher.py``: To run a single teacher-student experiment
- ``run_multiple_student_teacher.py``: To run an experiment with a single teacher and many students
- ``run_multiple_student_diff_sparsity.py``: To run an experiment with a single teacher and many students with different levels if sparsity