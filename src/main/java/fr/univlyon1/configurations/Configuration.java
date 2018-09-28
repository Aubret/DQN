package fr.univlyon1.configurations;

import lombok.Getter;
import lombok.Setter;

import javax.xml.bind.annotation.*;
import java.util.ArrayList;

@Getter
@Setter
@XmlRootElement(name="configuration")
@XmlAccessorType(XmlAccessType.FIELD)
public class Configuration {
    @XmlElement(name="epochs")
    int epochs = 50; // Nombre de time step avant la fusion des approximateurs
    @XmlElement(name="iterations")
    int iterations = 1; // Nombre d'itération d'apprentissage à chaque timestep
    @XmlElement(name="learn")
    int learn = 1; // NOmbre d'action avant chaque apprentissage
    @XmlElement(name="batchSize")
    int batchSize = 1; // Taille des batchs pour l'apprentissage

    //Paramètres de politiques
    @XmlElement(name="minEpsilon")
    Double minEpsilon = 0.01; // epsilon minimum dans epsilon decrmental
    @XmlElement(name="stepEpsilon")
    int stepEpsilon = 100; // nomre de pas pour décrémenter epsilon
    @XmlElement(name="noisyGreedyStd")
    Double noisyGreedyStd = 0.2 ;
    @XmlElement(name="noisyGreedyMean")
    Double noisyGreedyMean = 0. ;
    @XmlElement(name="initStdEpsilon")
    int initStdEpsilon = 2 ;


    // actor network
    @XmlElement(name="numHiddenNodes")
    int numHiddenNodes = 10; // Nombre de couches cachées
    @XmlElement(name="numLayers")
    int numLayers = 1; // Nombre de couches
    @XmlElement(name="learning_rate")
    Double learning_rate = 0.001; // Pas d'apprentissage

    @XmlElement(name="layersHiddenNodes")
    @XmlList
    ArrayList<Integer> layersHiddenNodes = new ArrayList<>();

    //critic network
    @XmlElement(name="numCriticHiddenNodes")
    int numCriticHiddenNodes = 10; // Nombre de couches cachées
    @XmlElement(name="numCriticLayers")
    int numCriticLayers = 1; // Nombre de couches
    @XmlElement(name="learning_rateCritic")
    Double learning_rateCritic = 0.001; // Pas d'apprentissage
    @XmlElement(name="criticHiddenNodes")
    @XmlList
    ArrayList<Integer> layersCriticHiddenNodes = new ArrayList<>();

    @XmlElement(name="numLstmHiddenNodes")
    int numLstmHiddenNodes = 10 ;
    @XmlElement(name="numLstmLayers")
    int numLstmlayers = 1;
    @XmlElement(name="learning_rateLstm")
    Double learning_rateLstm = 0.001 ;
    @XmlElement(name="LstmHiddenNodes")
    @XmlList
    ArrayList<Integer> layersLstmHiddenNodes = new ArrayList<>();
    @XmlElement(name="numLstmOutputNodes")
    int numLstmOutputNodes = 50 ;

    @XmlElement(name="forwardTime")
    int forwardTime = 10;
    @XmlElement(name="backpropTime")
    int backpropTime = 10 ;


    //expererience replay
    @XmlElement(name="sizeExperienceReplay")
    int sizeExperienceReplay = 5000; // Taille du buffer de l'expérience replay
    //TD
    @XmlElement(name="gamma")
    Double gamma = 0.9 ;

    @XmlElement(name="file")//Fichier de stockage mémoire experience replay
    @XmlList
    ArrayList<String> file = new ArrayList<>() ;

    public Configuration(){
    }

}
