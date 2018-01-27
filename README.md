
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