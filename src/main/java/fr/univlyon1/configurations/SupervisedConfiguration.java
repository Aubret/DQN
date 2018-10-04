package fr.univlyon1.configurations;

import lombok.Getter;
import lombok.Setter;

import javax.xml.bind.annotation.*;
import java.util.ArrayList;

@Getter
@Setter
@XmlRootElement(name="SupervisedConfiguration")
@XmlAccessorType(XmlAccessType.FIELD)
public class SupervisedConfiguration {
    @XmlElement(name="epochs")
    int epochs = 50; // Nombre de time step avant la fusion des approximateurs
    @XmlElement(name="iterations")
    int iterations = 1; // Nombre d'itération d'apprentissage à chaque timestep
    @XmlElement(name="learn")
    int learn = 1; // NOmbre d'action avant chaque apprentissage
    @XmlElement(name="batchSize")
    int batchSize = 1; // Taille des batchs pour l'apprentissage


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

    @XmlElement(name="timeDifficulty")
    int timeDifficulty = 1;

    @XmlElement(name="numberMaxInputs")
    int numberMaxInputs = 200 ;

    @XmlElement(name="dataBuilder")
    String dataBuilder = "DataList";

    @XmlElement(name="readfile")//Fichier de stockage mémoire experience replay
    @XmlList
    ArrayList<String> readfile = new ArrayList<>() ;

    @XmlElement(name="writefile")
    String writefile = "";

}
