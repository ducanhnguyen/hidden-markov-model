# hidden-markov-model
Hidden markov model

<b>Definition of hidden markov model</b>
<img src="https://github.com/ducanhnguyen/hidden-markov-model/blob/master/img/hmm_definition.png" width="750">

<b>Example of hidden markov model</b>
<img src="https://github.com/ducanhnguyen/hidden-markov-model/blob/master/img/hmm_example.png" width="750">

### Markov model case: Poem composer

Example of a poem generated by markov model. The markov model is trained on the poems of two authors: Nguyen Du (Truyen Kieu poem) and Nguyen Binh (>= 50 poems).

<img src="https://github.com/ducanhnguyen/hidden-markov-model/blob/master/img/poem_composer1.png" width="750">


## Train HMM for a sequence of discrete observations

Problem: Given a sequence of discrete observations, train a HMM

Link tutorial: <a href="https://web.stanford.edu/~jurafsky/slp3/A.pdf">HMM (standford)</a>

I just implemented this tutorial without any further optimizations.

<img src="https://github.com/ducanhnguyen/hidden-markov-model/blob/master/img/hmm_train.png" width="750">

In order to check the correctness of HMM, I use the following tests.

<table>
  <tr>
    <td>
      <b>Test 0</b>: we observe a sequence of 'H' and 'T' interchangeably.<br/>
      <img src="https://github.com/ducanhnguyen/hidden-markov-model/blob/master/img/hmm_test0.png" width="300"><br/>
      sequence = "THTHTHTHTHTHTHTHTHTHTHTHTHTHTHTHTHTHTHTH"
    </td>
    <td>
      <b>Test 1</b>: we only observe a sequence of tails.<br/>
      <img src="https://github.com/ducanhnguyen/hidden-markov-model/blob/master/img/hmm_test1.png" width="250">
    </td>
  </tr>
</table>

<table>
  <tr>
    <td>
      <b>Test 2</b>: we only observe a sequence of heads. </br>
      <img src="https://github.com/ducanhnguyen/hidden-markov-model/blob/master/img/hmm_test2.png" width="250"><br/>
      sequence = "HHHHHHHHHHHHHHHHHHHHHHHHHHH"
    </td>
    <td>
      <b>Test 3</b>: we observe a sequence of heads first, followed by a sequence of tails. </br>
      <img src="https://github.com/ducanhnguyen/hidden-markov-model/blob/master/img/hmm_test3.png" width="350"></br>
      sequence = "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT<br/>HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH"
    </td>
  </tr>
</table>

## Train HMM for a sequence of discrete observations (tf version)

Although I implemented HMM for a sequence of discrete observation based on the original algorithm, I still continue with a new way: facilitate the ability of gradient computation provided by TensorFlow.

The method contains two main steps:

- Step1: Define the loss function.

Here, the loss is a negative likelihood. We usually try to maximize the likelihood, which is known as the maximum likelihood estimation (MAE). In other words, we would minimize the negative likelihood.

- Step 2: Choose an adaptive learning rate optimizer such as Adam, then train the model under a number of iterations.

Results: A, B, pi (which minimize the loss)

## Train HMM for multiple sequences of discrete observations

This is the raw implementation based on <a href="https://www.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf">this tutorial</a>

In order to check the correctness of HMM, I use the following tests.

<b>Test 0</b>. Consider two sequences. 

Sequence 1 = "TTTTTTTTTTTTTTTTTTTTTTT"

Sequence 2= "HHHHHHHHHHHHHHHHHHHHHHHH"

<table>
  <tr>
    <td>
      <img src="https://github.com/ducanhnguyen/hidden-markov-model/blob/master/img/hmm_test4.png" width="350">
    </td>
    <td>
      <img src="https://github.com/ducanhnguyen/hidden-markov-model/blob/master/img/hmm_test4_loglikelihood.png" width="550">
    </td>
  </tr>
 </table>
 
<b>Test 1</b>. Consider two sequences. 

Sequence 1 = "THTHTHTHTHTHTHTHTHTHTHTHTH"

Sequence 2= "THTHTHTHTHTHTHTH"
  
<table>
  <tr>
    <td>
      <img src="https://github.com/ducanhnguyen/hidden-markov-model/blob/master/img/hmm_test5.png" width="350">
    </td>
    <td>
      <img src="https://github.com/ducanhnguyen/hidden-markov-model/blob/master/img/hmm_test5_loglikelihood.png" width="550">
    </td>
  </tr>
 </table>

## Train HMM for multiple sequences of discrete observations (tf version)

In this part, I will train HMM using TensorFlow (i.e., not use the original algorithm).

<b>Experiment</b>

Training

- Sequence 1: TTTTTTTTTTTTTTTTTTTTTTTTTTTHTTTT

- Sequence 2: HHHHHHHHHHHHHHHTHHHHHHHHH

Testing: 

- Expectation: The probability of test 2 and 3 should be very high. The probability of test 1, 4, 5 should be very low. The probability of test 4 should be higher than that of test 5.

- Real result: The result is the same as what we are expected.

| id | Test | Probability | Log probability |
|  -------------  | ------------- | ------------- | ------------- |
| 1 |  THTHTHTHTHTHTHTHTHTHTH |  4.723788581964585e-16 | -35.28875034460062 |
| 2 |  TTTTTTTTTTTTTTTTTTTTTT |  0.8191389207730484 |-0.1995015870798115|
| 3 |  HHHHHHHHHHHHHHHHHHHHHH |  0.6648675313118346 |-0.40816745920427944|
| 4 |  TTTTTTTTTHTTTTTTTTTTTT |  0.02642439012907365 | -3.633467826808207|
| 5 | TTTTTTTTTHTTTTTTHTTTTTT  |  0.0008257384866132875 | -7.099232436735432|
