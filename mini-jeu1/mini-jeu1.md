https://teachablemachine.withgoogle.com/

**GOAL**: Entrainer un modèle à dire quelle peluche il voit quand on lui en montre une.

## **Step 01**: Prendre des photos des deux peluches
Environ 5 à 10 photos de chaques.

- **Problème**: Arrive pas moins bien à détecter les peluches sous des angles différents

## **Step 02**: Ajouter des photos des peluches dans tous les angles

    /!\ Déséquilibrer les photos volontairement

- **Problème**: classes déséquilibrées; biaisée.

## **step 03**: ré-équilibrer

- **Problème**: Toujours des probabilités de l'une des 2 classes quand on à aucune des deux peluches à l'image.


## **Step 05**: Ajouter une class null avec que le contexte.



## **Questions à aborder tout au long**

- Comment nommer une classe
- Qu'est-ce que ca veut dire 90% en score (probabilité/certitude)
- Qu'elles sont les caractéristiques qui permettent la discrimination (utilisation d'autres objets qui ont la meme couleur/forme/contexte biaisé)
- Qu'est-ce qu'un biais 
