# hidden-markov-model
Hidden markov model

<b>Definition of hidden markov model</b>
<img src="https://github.com/ducanhnguyen/hidden-markov-model/blob/master/img/hmm_definition.png" width="750">

<b>Example of hidden markov model</b>
<img src="https://github.com/ducanhnguyen/hidden-markov-model/blob/master/img/hmm_example.png" width="750">

### Train HMM (forward-backward algorithm)

Problem: Given a sequence of observations, train HMM

Link tutorial: <a href="https://web.stanford.edu/~jurafsky/slp3/A.pdf">HMM (standford)</a>

I just implemented this tutorial without any further optimizations.

<img src="https://github.com/ducanhnguyen/hidden-markov-model/blob/master/img/hmm_train.png" width="750">


In order to check the correctness of HMM, I use the following tests.

<b>Test 0: we observe a sequence of 'H' and 'T' interchangeably. </b>
  
<img src="https://github.com/ducanhnguyen/hidden-markov-model/blob/master/img/hmm-test0.png" width="450">

<b> Test 1: we only observe a sequence of tails. </b>
  
<img src="https://github.com/ducanhnguyen/hidden-markov-model/blob/master/img/hmm-test1.png" width="450">

<b> Test 1: we only observe a sequence of heads. </b>

<img src="https://github.com/ducanhnguyen/hidden-markov-model/blob/master/img/hmm-test2.png" width="450">
