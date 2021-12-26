from imageai.Classification import ImageClassification
models = ImageClassification()

models.setModelTypeAsResNet50()

models.setModelPath("resnet50_imagenet_tf.2.0.h5")

models.loadModel()

predictions, probabilities = models.classifyImage("aaaaaaaaaaa.jpg", result_count=20)

for i in range(len(predictions)):
    print(f"{predictions[i]} : {probabilities[i]}")
