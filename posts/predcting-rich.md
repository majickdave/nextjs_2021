---
title: "Predicting Rich Attributes in Real Estate Images Using Fastai"

date: "2019-02-05"
---

# Overview
Visual attribute search can greatly improve the user experience, and SEO for home listing and travel websites. Although Zillow, Redfin, Airbnb and TripAdvisor have some metadata already about the amenities of a property, they can expand searchable attributes by analyzing the property images with vision models.
In this post, we share our initial approach towards a few-shot model for predicting property attributes like view, kitchen island, pool, high ceilings, hardwood floors, fireplace, etc. Since these attributes are often room and context dependent, we begin with an accurate classification model to group our images into interior and exterior settings of the property.
In the process of training our initial room-type model, we notice that some of these rich attributes are easily separable in [platform.ai](http://platform.ai/)

# Background
Previous work has focused on using images to improve price estimation [1] however the incremental gain of adding image features to pricing models has been minimal; 2.3% improvement over using a handful of conventional attributes like location and property size. While pricing data for building these models has been readily available, there is a scarcity of datasets for predicting rich attributes like view, kitchen island, pool, high ceilings, hardwood floors, fireplace, etc.
Our initial dataset, previously used in price estimation [1], consists of 146,791 images and seven classes: living room, dining room, bedroom, bathroom, kitchen, interior and exterior.

![barchart count of features](https://drive.google.com/uc?export=download&id=119Uv2pYBrqpQG5nYf2L77guBzadJdae6 "Fig 1. Class count of real estate images")

Bathroom is the most underrepresented class, with nearly half the amount of images than any other class. We addressed this class imbalance using [fastai](http://fast.ai/)'s vision.transform method [4] to oversample the data using the default image augmentations.

![training images](https://drive.google.com/uc?export=download&id=1wHF_jPqhdYWvf1xotJGRGeDj40z63eQW "Fig 2. Example image augmentation of the classes: bathroom, dining room, kitchen, living room, bedroom, interior, and exterior.")

The images were pre-processed using fast.ai’s built-in transforms. Data was split randomly into 60% train, 20% validation and 20% test.
The model was initialized with ImageNet-ResNet34 weights. The network’s custom head was trained for 3 epochs, followed by unfreezing the entire network and fine tuning for another 10 epochs using discriminative learning rates. Fine tuning improved the model fit, achieving an overall test set accuracy: 97%.
By increasing the network capacity to a ResNet50, 98% final accuracy was achieved — a significant improvement over the 91% accuracy of the previous results [1].

## Building a Rich Attribute Dataset
We constructed a rich attribute dataset by crawling property listing websites. The crawler captured both images, and the attributes of interest. In total 18,790 listings were obtained along with 350,000 images.

# Feature Class Distribution
Our web scraper captured unstructured html, and extracted the rich attributes contained in the listings’ details table.

![sample post](https://drive.google.com/uc?export=download&id=1okD6rggR9Ty5gnfhiBWRQEcE5K8n7whg "Fig 3. Example scraped listing text metadata")

![barchart ratio of features](https://drive.google.com/uc?export=download&id=1E_lE4Kn1k6WYSzroR9pgmh4FNlxMejIa "Figure 4. Feature class distribution from crawled data")

The final data set consists of 18,790 individual listings that each hold an average of 21 images. We have identified several features visible in the photos like pools, patios, kitchen islands, and fireplaces. Nearly half of the listings in our data set have a pool or a patio, and only about 25 listings have wine cellars. Furthermore, the appearance of the attribute can be seen in different spaces; modern wine cellars tend to be above ground.


![wine cellar 1](https://drive.google.com/uc?export=download&id=1aSv00WtD0YRDbm7aAu3PjaraMSoigfDP "Fig 5a. Example feature from listings dataset: wine cellar")

![wine cellar 2](https://drive.google.com/uc?export=download&id=1_ci-u8zshSZWUVLaWoUCWF-tlZS8h17I "Fig 5b. Example feature from listings dataset: wine cellar")

![wine cellar 3](https://drive.google.com/uc?export=download&id=1tdrv1bXnp1fU8WKMe--ldlEJyBNhfvXk "Fig 5c. Example feature from listings dataset: wine cellar")

![wine cellar 4](https://drive.google.com/uc?export=download&id=1ySQ79Xa2Pssx9J2DgujV-m-u0OnQjhgn "Fig 5d. Example feature from listings dataset: wine cellar")

# Projections
We uploaded our model, and a sample of 20,000 images from our dataset to [platform.ai](http://platform.ai/) in order to compare its performance against the pre-built ImageNet model. Our model forms neat clusters, easily separable by eye, of similar attributes of interest like fireplaces, pools, and kitchen islands. In comparison, ImageNet tends to form wider clusters with dissimilar attributes.

![Our Model's Projection](https://drive.google.com/uc?export=download&id=1kU-9CCg0i_Y8dLiJnXJqLVUkPhxUBM5h "Fig 6. Pictured: Our Model’s Projection")

![ImageNet Projection](https://drive.google.com/uc?export=download&id=1AhPOn27gEIjr7bHNaPQ_7jeXPaXhD89s "Fig 7. Pictured: ImageNet Projection")

![platform.ai projection1](https://drive.google.com/uc?export=download&id=1tfzDzwhrAqmuYedCCqF1bi7Gzjk3fMLL "Fig 8. Zoomed in projections show a fireplace cluster.")

![platform.ai projection2](https://drive.google.com/uc?export=download&id=162gYC0VVdKBgolRwePGXxvThn5z2BdvB "Fig 9. Zoomed in projections show a kitchen islands cluster.")

![platform.ai projection3](https://drive.google.com/uc?export=download&id=1UJv7bVvfTck_Wl4xXZh4-EjuC6_NnV4n "Fig 10. Zoomed in Projections, and selected images from our model show an outdoor swimming pool cluster.")

Using the projections as visual aids, clusters of interest were highlighted, and selectively filtered using platform.ai. The zoomed in views of our model projection show three rich features which we have identified through our model: fireplace, kitchen island, and pool. When compared with ImageNet, we can see more numerous clusters bound closely to rich attributes vs. labeled room class features.

# Cluster Analysis
After downloading our projections, we were able to evaluate a clustering solution comparing our model’s silhouette score against ImageNet. The results show that our silhouette score is significantly greater than ImageNet per t-test results on k=5 K-means clusters Silhouette score. Thus, our model produces similar clusters more consistently than ImageNet-ResNet.

![cluster analysis](https://drive.google.com/uc?export=download&id=1WyPlGjC8uPWbq2S4aNKjpjtpJw8nRzbG "Fig 9. Similarity “Silhouette” scores for k=5 K-Means clusters.")

![silhouette score](https://drive.google.com/uc?export=download&id=1RHcdWyxi3Ugb3ccVSh8J4VB5YuzwbHjQ "Table I. Silhouette Score summary statistics")

# Conclusion
Applying modern machine learning practices, we have developed a computer vision model that not only predicts room classes, but also the deeper attributes present in the homes that we live in. It performed better than ImageNet by clustering our nested attributes closer together, allowing visually separable groups to be extracted and labeled. Development of an accurate attribute search model could be implemented as an essential search tool in finding the right home or rental.
We plan on developing our model further using the limited labeled data from our dataset, and a Relation Network (RN) [2] to classify multiple attributes in images.

## Acknowledgements
We’d like to thank Arshak Navruzyan for his mentor support and guidance during this project. We would also like to thank fastai team for a convenient deep learning library.

## References

1. Poursaeed, Omid et al. [Vision-based real estate price estimation](https://omidpoursaeed.github.io/publication/vision-based-real-estate-price-estimation/). Machine Vision and Applications 29 (2018): 667–676.
2. Santoro, Adam et al. [A simple neural network module for relational reasoning](https://arxiv.org/abs/1706.01427). NIPS (2017).
3. He, Kaiming et al. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2016): 770–778.
4. Howard, Jeremy, et al. [fastai library](https://docs.fast.ai/). 2019.
5. Clarke, Adrian, et al. [Optimizing hyperparams for image datasets in fastai](https://platform.ai/blog/page/1/optimizing-hyperparams-for-image-datasets-in-fastai/). 2019