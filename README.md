# Machine learning for engineers mini-project
## Metal classifier Group N°5 Ghazraoui-Graf

### procédure pour tester des pièces de monnaies :

    pour faire un test live avec l'appareil (ISS) télécharger ISS.exe et lancer le notebook "live_test.ipynb" puis suivre les instructions
    pour tester directement un ou plusieurs fichier .h5 lancer le notebook "live_files.ipynb" puis suivre les instructions
    
### données:

Les données utilisées pour l'entrainement des modèles : /data/Groupe5-8-11
Les données de pièces étrangère utilisées pour les testset : /data/foreign
Les modèles entrainés : /models

### les scripts utilisé pour entrainer les modèles:

    train_anomaly.ipynb (pas utilisé)
    train_SVM_AGF.ipynb (modèle de classificaiton CHF)
    train_SVM_onevsall.ipynb (one class svm pour la détection de pièces étrangères-frauduleuses)