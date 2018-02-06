
# Projet MathAppli 2017 - 2018
# Deep Networks for Character Recognition

Ce projet traite de l'utilisation de l'utilisation de *deep networks* pour la 
reconnaissance de caractères. On se restreint ici en fait à la reconnaissance 
de chiffre et on utilisera principalement le jeu de données MNIST. 
Le langage Python sera utilisé pour coder l'écriture des programmes et des 
algorithmes.

> Le jeu de données étant un peu lourd pour un repo GitHub, nous vous invitons à 
> le télécharger par vous même [ici](http://yann.lecun.com/exdb/mnist/) (site de 
> Yann LeCun) ou à utiliser la bibliothèque Python `keras` qui contient déjà les 
> données.

Plusieurs techniques de reconnaissance ont été mises en place.


## Reconnaissance avec un réseau de neurones simple

Le fichier `network.py` implémente un réseau de neurones utilisant la fonction 
d'activation sigmoïde et la méthode de la descente de gradient stochastique 
pour l'apprentissage. 

Chaque image du jeu d'entraînement est redimenssionnée en un vecteur de 784 réels 
compris entre 0 et 1. Ce vecteur est donné directement en entré du réseau. 
La sortie du réseau est un vecteur de taille 10, correspondant aux 10 chiffres 
que l'on souhaite reconnaître.

En choisissant d'utiliser une seule couche cachée de 30 neurones et un taux 
d'apprentissage de `3.0`, on peut obtenir un taux de reconnaissance de l'ordre 
de 95% sur le jeu de test. 


## Reconnaissance basée sur l'extraction de *features*

Cette technique consiste à extraire des images des données statistiques ou 
structurelles appellées *features* et à utiliser ces *features* comme entrées 
de réseaux de neurones.

Nous avons implémenter plusieurs *features*, parmi lesquelles:
- le nombre de boucles ;
- les densités de pixels par zones ;
- l'extraction de contour ;
- transformation de Fourier de l'image.

Le calcul de ces *features* est effectué dans le fichier `features.py`.
Ces calculs prennent un temps important, c'est pourquoi un système de cache 
a été mis en place. Cepedant, comme le volume de données générées est important, 
ce cache n'est pas complètement présent dans le repo.

### Calcul des *features*

Le calcul des *features* se fait en executant le script `features.py`:

```batch
python features.py
```

Le temps de calcul est assez long (~ 1 heure), mais n'est à effectuer 
qu'une seule fois. Après quoi, les données sont écrites dans le répertoire 
`cache`.


## Reconnaissance avec des réseaux convolutifs

Le fichier `cnn.py` implémente un réseau convolutifs en utilisant la 
bibliothèque Python `keras`.
Les données fournies en entrées sont similaires à celles utilisées 
pour le réseau simple (pas d'utilisation de *features*). 

On obtient avec cette méthode un taux de reconnaissance de l'ordre 
de 99%.
