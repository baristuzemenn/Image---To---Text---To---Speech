<img width="252" alt="image" src="https://github.com/user-attachments/assets/7e6e3499-46ca-4434-bcc7-24d912102e8a"><img width="252" alt="image" src="https://github.com/user-attachments/assets/93b1bd0e-9afa-4643-bb31-d2b00eac607c"><img width="250" alt="image" src="https://github.com/user-attachments/assets/c832b49b-d3f0-4c67-8bb4-85dc04f846d5">
<img width="252" alt="image" src="https://github.com/user-attachments/assets/c07e4f60-64eb-4d94-ba8c-339632af00ba">



1. Aim of the Study
The objective of this study is to enhance media accessibility by developing a sophisticated system that automatically generates textual descriptions of images and converts these descriptions into spoken words. This project aims to make visual content accessible to a broader audience, including individuals with visual impairments. By integrating image captioning using Vision Transformer (ViT) and GPT-2 with TTS technology using Tacotron2, the system will provide an inclusive media experience that allows users to "see" through hearing.
2. Models and Algorithms
2.1. Model Description
The system comprises two primary components:
Image Captioning: Utilizes the Vision Transformer (ViT) for extracting detailed features from images. ViT applies the transformer architecture, typically used in natural language processing, to the realm of image analysis. The extracted features are then processed by GPT-2, a powerful text generation model, which produces captions that describe the images in a contextually relevant manner.
Text-to-Speech (TTS): Employs Tacotron2, an advanced speech synthesis model. Tacotron2 transforms the text captions into lifelike spoken audio, mimicking human speech patterns. This part of the system ensures that the generated descriptions are delivered in a clear and natural-sounding voice.
2.2. Algorithm
Image Captioning Workflow: Begins with preprocessing where images are resized and normalized to fit the input requirements of ViT. The ViT model processes these images and outputs embeddings that represent visual features. These embeddings serve as input for GPT-2, which generates coherent and contextually appropriate captions.
TTS Workflow: The captions are fed into Tacotron2, which first converts the text into a spectrogram, representing how the speech should sound. The spectrogram is then converted into audible waveforms through a vocoder, producing natural-sounding speech.
2.3. Additional Techniques
Data Augmentation: To enhance the robustness of the image captioning model, data augmentation techniques such as rotation, scaling, and color adjustment are applied to the training images.
Attention Mechanisms: In both the image captioning and TTS processes, attention mechanisms are used to focus on relevant features and text segments, improving the accuracy and relevance of the output.
Normalization: For TTS, normalization of text involves converting all characters to a uniform case, expanding abbreviations, and handling special characters, which helps in maintaining consistency in speech output.
3. Data Set
3.1. Features
Image Captioning: The dataset used is Flickr30k, which contains 30,000 images accompanied by five different captions each. This dataset provides a diverse set of images from various contexts, which helps in training the model to handle a wide range of visual scenarios.
TTS: The dataset consists of audio files and their corresponding textual transcripts sourced from Kaggle. This dataset is specifically curated to train TTS models, ensuring that the speech output is as natural and varied as possible.
3.2. Data Preparation and Preprocessing
Image Data: Images are resized to 224x224 pixels to match the input size of ViT. Each image is normalized to have pixel values between 0 and 1, aligning with the neural network's expected input range.
Text Data for Captioning: Captions are tokenized using the tokenizer from GPT-2, which converts text into a sequence of integers. Tokenization helps in processing variable-length text data efficiently.
Audio Data for TTS: Audio files are resampled to a standard sampling rate (commonly 22,050 Hz), and the corresponding texts are cleaned and preprocessed to remove any non-standard speech patterns that could affect training.
4. Analysis
4.1. Implementation of the Model
Image Captioning Model: The CaptioningModel class integrates the ViT and GPT-2 models. ViT acts as the feature extractor, capturing complex visual patterns from the input images, which are resized and normalized as per the model's requirements. The output features of ViT are then fed into GPT-2, which generates the captions. This model uses a custom forward method that combines the features from ViT with the embedding layer of GPT-2 to produce coherent captions.
Text-to-Speech Model: The SimpleTacotron2 class implements a basic version of the Tacotron2 architecture. It processes the text obtained from the image captioning model and generates a mel spectrogram, which is later converted into audible speech. The model primarily utilizes an LSTM network to capture the temporal dependencies in the input text, followed by a fully connected layer that predicts the audio output.
4.2. Training of the Model with the Algorithm
The training process involves feeding the processed data into the models and optimizing their parameters:
Training the Image Captioning Model: The training loop loads batches of images and their corresponding captions. The images are passed through the ViT and GPT-2 model sequence to generate predicted captions. The output is then compared to the actual captions using cross-entropy loss, which measures the discrepancy between the predicted and true captions. This loss is minimized using the Adam optimizer.
Training the TTS Model: The Tacotron2 model is trained using pairs of text captions and their corresponding audio files. The model predicts the audio from the text and the loss is calculated as the mean squared error between the predicted and actual audio waveforms. This loss is also minimized using the Adam optimizer.
4.3. Search for Meta Parameters
Image Captioning: Key parameters include the learning rate, batch size, and number of epochs. These are adjusted based on the performance of the model on the validation set, ensuring that the model neither overfits nor underfits.
TTS: Similar to the image captioning model, the learning rate, batch size, and number of training epochs are tuned. Additionally, the architecture-specific parameters like the number of LSTM units and layers in the Tacotron2 model are also optimized based on the validation loss.
4.4. 5-Fold Cross Validation Results
To evaluate the robustness and generalizability of the models, 5-fold cross-validation is employed:
Image Captioning: The dataset is split into five folds. In each iteration, one fold is used as the validation set while the others are used for training. This process helps in assessing how well the model performs across different subsets of the data.
TTS: The cross-validation process for the TTS model follows a similar structure, ensuring that the model's ability to generate speech is consistently evaluated across different subsets of text and corresponding audio files.
The results from the cross-validation provide insights into the models' performance, highlighting their effectiveness in generating accurate captions and speech outputs. Metrics such as BLEU score for captioning and 
