# Benchmarking-Results-of-Applying-Positive-Unlabeled-Learning-to-the-Medical-Domain

The experiments are carried out by three implementations in pairwise, uPU [1] and nnPU [2], nnPU and ImbalancednnPU [3]. This is a reproducing code for the project.

Requirements:
1. Python
2. Numpy
3. Chainer
4. PyTorch
5. Scikit-learn
6. Matplotlib

Example of code execuation is in run.sh.

Databases hyper-links:

MIT-BIH: https://physionet.org/content/mitdb/1.0.0/

AHA: https://physionet.org/content/ahadb/1.0.0/

European ST-T: https://physionet.org/content/edb/1.0.0/

PTB-XL: https://physionet.org/content/ptb-xl/1.0.1/

ECG Signal:

![jmyu](https://github.com/Blacksite-2008/Benchmarking-Results-of-Applying-Positive-Unlabeled-Learning-to-the-Medical-Domain/assets/53436099/aa773a77-b57c-4d22-bef5-fa47732523b1)

Example Result:

After running MIT-BIH on Training with uPU and nnPU. The errors are measured by zero-one loss.

<img width="433" alt="asfa" src="https://github.com/Blacksite-2008/Benchmarking-Results-of-Applying-Positive-Unlabeled-Learning-to-the-Medical-Domain/assets/53436099/400024ea-3ec7-4e25-a252-a70e92bf2794">

Reference:

[1] M. C. Du Plessis, G. Niu, and M. Sugiyama. Analysis of learning from positive and unlabeled data. Advances in neural information processing systems, 27, 2014.

[2] R. Kiryo, G. Niu, M. C. Du Plessis, and M. Sugiyama. Positive-unlabeled learning with nonnegative risk estimator. Advances in neural information processing systems, 30, 2017.

[3] G. Su, W. Chen, and M. Xu. Positive-unlabeled learning from imbalanced data. In Proceedings of the 30th International Joint Conference on Artificial Intelligence, Virtual Event, 2021.

