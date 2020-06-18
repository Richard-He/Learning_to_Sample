###Learning to Sample

 

####MetaModel Input:

1.Best Valid Loss at V_sample

2.Average Training Loss for the Past 5 time

3.Probability for Each Class

4.Label for each class 

TODO:

- [X] Change the actual value to the rank
- [ ] Loss Function
- [ ] Soft Decision
 

 

####Decision:

Prob>0.5 choose

Prob<0.5 drop

 

####Meta_loss

Prob*Valid_Loss

 

####Issue:

Variance too small