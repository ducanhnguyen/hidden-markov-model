# hidden-markov-model
Hidden markov model

<b>Definition of hidden markov model</b>
<img src="https://github.com/ducanhnguyen/hidden-markov-model/blob/master/img/hmm_definition.png" width="750">

<b>Example of hidden markov model</b>
<img src="https://github.com/ducanhnguyen/hidden-markov-model/blob/master/img/hmm_example.png" width="750">

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

## Train HMM for multiple sequences of discrete observations

<a href="https://www.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf">Tutorial</a>

<b>Test 0</b>. Two sequences. Sequence 1 = "TTTTTTTTTTTTTTTTTTTTTTT". Sequence 2= "HHHHHHHHHHHHHHHHHHHHHHHH"

<table>
  <tr>
    <td>
      <img src="https://github.com/ducanhnguyen/hidden-markov-model/blob/master/img/hmm_test4.png" width="350">
    </td>
    <td>
      <img src="https://github.com/ducanhnguyen/hidden-markov-model/blob/master/img/hmm_test4_likelihood.png" width="350">
    </td>
  </tr>
 </table>
 
<b>Test 1</b>. Two sequences. Sequence 1 = "THTHTHTHTHTHTHTHTHTHTHTHTH". Sequence 2= "THTHTHTHTHTHTHTH"
  
<table>
  <tr>
    <td>
      <img src="https://github.com/ducanhnguyen/hidden-markov-model/blob/master/img/hmm_test5.png" width="350">
    </td>
    <td>
      <img src="https://github.com/ducanhnguyen/hidden-markov-model/blob/master/img/hmm_test5_likelihood.png" width="350">
    </td>
  </tr>
 </table>
