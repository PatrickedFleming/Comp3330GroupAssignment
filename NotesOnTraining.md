# Trying Different Learning Rates
`optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001) #still going down,`
```
Epoch: 25, train accuracy: 36.09%, train loss: 1.8014
Epoch: 25, val accuracy: 6.90%, val loss: 3.2368
Test loss: 4.0203
Test accuracy: 6.90%
```

`optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01) #FLATLINED EARLY`
```
Epoch: 25, train accuracy: 18.12%, train loss: 2.9532
Epoch: 25, val accuracy: 10.34%, val loss: 3.8983
Test loss: 4.2494
Test accuracy: 0.00%
```

`optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001) #STILL GOING DOWN`
```
Epoch: 25, val accuracy: 3.45%, val loss: 3.4849
Test loss: 3.7415
Test accuracy: 3.45%
```
# Trying 5th layer with 256 nodes
```
Epoch: 25, train accuracy: 57.51%, train loss: 1.4803
Epoch: 25, val accuracy: 13.79%, val loss: 3.2000
Test loss: 3.6078
Test accuracy: 13.79%
```
Much better accuracy.

# Trying 6th Layer
```
Epoch: 25, val accuracy: 13.79%, val loss: 3.1940
Test loss: 3.8979
Test accuracy: 6.90%
```

after epoch 250:
```
Epoch: 250, train accuracy: 98.44%, train loss: 0.0515
Epoch: 250, val accuracy: 24.14%, val loss: 4.4472
Test loss: 5.4258
Test accuracy: 24.14%
```

appeared to flatline after 50 epoch, so going to decrease learning rate, try again..
