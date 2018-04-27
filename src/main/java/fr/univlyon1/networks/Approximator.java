package fr.univlyon1.networks;

import fr.univlyon1.actorcritic.policy.Policy;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface Approximator extends Policy{
    void init(); // Initialisation
    INDArray getOneResult(INDArray data); // Obtenir un résultat
    int numOutput(); // taille de l'INDArray résultant
    Object learn(INDArray input,INDArray labels,int number); // Apprentissage supervisé
    INDArray error(INDArray input,INDArray labels,int number); // retourne l'erreur sur l'entrée sans apprentissage
    INDArray getScoreArray();


    INDArray getParams() ; // Permet de dupliquer les paramètres notamment
    void setParams(INDArray params) ;


    Approximator clone(); // clônage
    Approximator clone(boolean Listener);
    void stop(); // Fin d'apprentissage
}
