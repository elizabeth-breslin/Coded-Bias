# Coded-Bias

## Contents

This Repository contains three main folders and two additional files. 
 - The SRC folder contains the main source code of our analysis
 - The DATA folder contains the data used in the analysis as well as documentation about the data
 - The LICENSE.md file explains to a visitor the terms under which they may use and cite this repository
 - This README.md file serves as an orientation to this repository.

## Hypothesis
If we train a model to classify face images as either male or female with only white faces as opposed to all racial groups, it will be at least 10% less accurate in classifaction of males and females for all racial groups in comparison to a model trained with images of all racial groups. 

## SRC Folder 

## Data Folder
### Data Collection Process:
We pulled our image data from the UTKFace dataset. The set contains over 20,000 people's faces; we narrowed our selection to 18-106 year olds. Each photo is labeled by age, gender, race, date, and time and are then separated into folders for race and gender in the Data Github folder. Python was used to download the faces, split the data into groups separated by gender and race, and finally upload to Github. Github truncated teh data the we uploaded: only 1000 photos are present in each folder and 385 are omitted. To obtain the full dataset for yourself, use Python as instructed (and look at MI2 for more in-depth explanations) and download the data from https://susanqq.github.io/UTKFace/.

### Data File:
<a href="https://github.com/elizabeth-breslin/Coded-Bias/blob/master/DATA/">DATA folder</a>

### Data Dictionary:
The Data folder contains one folder for each gender/race combination we are exploring. Combinations include:
Asian/male, Asian/female, Indian/male, Indian/female, Black/male, Black/female, White/male, White/female, Other/male, Other/female

## Figures Folder
### allRace_Regression:



## References
### Prepatory Assignments: 
<a href="https://github.com/elizabeth-breslin/Coded-Bias/blob/efc06f893a139fd6fa35b74cafa8dc1b5a5d33ab/MI1-3.pdf">M1: Hypothesis</a> <br>
<a href="https://github.com/elizabeth-breslin/Coded-Bias/blob/423f558c77c18c39c59febf9dbcbacd094c97b0f/MI2.pdf">M2: Establish Data</a>

### Acknowledgements: 
Thanks to Professor Alonzi for his guidance, wisdom, and inspiration and thanks to Harsh for his expertise each step of the way. <br> 

### References: 
[1]	B. Rauenzahn, J. Chung, A. Kaufman, “Facing Bias in Facial Recognition Technology” The Regulatory Review, Mar, 20 2021. [Online]. Available: https://www.theregreview.org/2021/03/20/saturday-seminar-facing-bias-in-facial-recognition-technology/#:~:text=According%20to%20the%20researchers%2C%20facial,particularly%20vulnerable%20to%20algorithmic%20bias. [Accessed:  Nov. 02 2022]

[2] 	A. Fawcett, “Understanding racial bias in machine learning algorithms” DEV, June, 8 2020. [Online]. Available: https://dev.to/educative/understanding-racial-bias-in-machine-learning-algorithms-4cij [Accessed. Nov, 02 2022]

[3]	Darshan M. “How to use logistic regression for image classification?”. Analytics Indian Magazine. June, 5 2022. [Online]. Available: https://analyticsindiamag.com/how-to-use-logistic-regression-for-image-classification/
	[Accessed: Nov, 02 2022]

[4]	G. Galario, “Image Classification using Logistic Regression on the AMerican Sign Language MNIST”. Medium. June, 5 2020. [Online]. Available: https://medium.com/@gryangalario/image-classification-using-logistic-regression-on-the-american-sign-language-mnist-9c6522242ddf [Accessed: Nov, 02 2022]
