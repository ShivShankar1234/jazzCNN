# jazzCNN
Jazz Music Generator

This project was built using PixelCNN++ architecure and the Varaitional Autoencoder philosophy.
Training utilized 20000 epochs with a dataset of Jazz Piano tracks in the midi file.
I worte much of the code used to parse the midi files and extract useful features (i.e. note duration, pitch) and convert it into a visual analytic pixel format that was utilized by the PixelCNN framework. 
Accuracy: 89%
Loss: 12%

Sources:
https://papers.nips.cc/paper/6527-conditional-image-generation-with-pixelcnn-decoders.pdf
https://github.com/openai/pixel-cnn
