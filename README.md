# DeepEvent Version 0.3
---
We propose a new application (_DeepEvent_) of long short term memory recurrent neural network to the automatic detection of gait events.
The 3D position and velocity of the markers on the heel, toe and lateral malleolus were used by the network to determine Foot Strike (FS) and Foot Off (FO). 
The method was developed from  10526 FS and 9375 FO from 226 children. _DeepEvent_ predicted **FS** within **5.5 ms** and **FO** within **10.7 ms** of the gold standard (automatic determination using force platform data) and was more accurate than common heuristic marker trajectory-based methods proposed in the literature and another deep learning method.

---
## Requirement
Linux, Python 2.7, Keras, Tensorflow, Btk, Numpy, Scipy.

---
## Installation
`pip install deepevent==0.2`

---
## Running _DeepEvent_
`deepevent FilenameIn.c3d FilenameOut.c3d`
where FilenameIn.c3d is the c3d file to identify gait events, FilenameOut.c3d is the new file with the gait events.

---
## Biliography
Lempereur M., Rousseau F., Rémy-Néris O., Pons C., Houx L., Quellec G., Brochard S. (2019). A new deep learning-based method for the detection of gait events in children with gait disorders: Proof-of-concept and concurrent validity. Journal of Biomechanics. Available online 9 November 2019. In Press, Corrected Proof. https://doi.org/10.1016/j.jbiomech.2019.109490.

---
## Next step
Python 3.7, Windows

---
## Contact
[mathieu.lempereur@univ-brest.fr](mailto:mathieu.lemepreur@univ-brest.fr)  
Mathieu LEMPEREUR  
CHRU de Brest - Hopital Morvan  
Service MPR  
2 avenue Foch  
29609 BREST cedex  
FRANCE