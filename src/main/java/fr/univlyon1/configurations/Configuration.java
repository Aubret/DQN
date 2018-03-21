package fr.univlyon1.configurations;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;

@XmlRootElement(name="configuration")
@XmlAccessorType(XmlAccessType.FIELD)
public class Configuration {
    @XmlElement(name="epochs")
    int epochs = 50; // Nombre de time step avant la fusion des approximateurs
    @XmlElement(name="iterations")
    int iterations = 1; // Nombre d'itération d'apprentissage à chaque timestep
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

    // actor network
    @XmlElement(name="numHiddenNodes")
    int numHiddenNodes = 10; // Nombre de couches cachées
    @XmlElement(name="numLayers")
    int numLayers = 1; // Nombre de couches
    @XmlElement(name="learning_rate")
    Double learning_rate = 0.001; // Pas d'apprentissage

    //critic network
    @XmlElement(name="numCriticHiddenNodes")
    int numCriticHiddenNodes = 10; // Nombre de couches cachées
    @XmlElement(name="numCriticLayers")
    int numCriticLayers = 1; // Nombre de couches
    @XmlElement(name="learning_rateCritic")
    Double learning_rateCritic = 0.001; // Pas d'apprentissage

    //expererience replay
    @XmlElement(name="sizeExperienceReplay")
    int sizeExperienceReplay = 5000; // Taille du buffer de l'expérience replay
    //TD
    @XmlElement(name="gamma")
    Double gamma = 0.9 ;

    public Configuration(){
    }

    public int getEpochs() {
        return epochs;
    }

    public void setEpochs(int epochs) {
        this.epochs = epochs;
    }

    public int getIterations() {
        return iterations;
    }

    public void setIterations(int iterations) {
        this.iterations = iterations;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    public Double getMinEpsilon() {
        return minEpsilon;
    }

    public void setMinEpsilon(Double minEpsilon) {
        this.minEpsilon = minEpsilon;
    }

    public int getStepEpsilon() {
        return stepEpsilon;
    }

    public void setStepEpsilon(int stepEpsilon) {
        this.stepEpsilon = stepEpsilon;
    }

    public int getNumHiddenNodes() {
        return numHiddenNodes;
    }

    public void setNumHiddenNodes(int numHiddenNodes) {
        this.numHiddenNodes = numHiddenNodes;
    }

    public int getNumLayers() {
        return numLayers;
    }

    public void setNumLayers(int numLayers) {
        this.numLayers = numLayers;
    }

    public Double getLearning_rate() {
        return learning_rate;
    }

    public void setLearning_rate(Double learning_rate) {
        this.learning_rate = learning_rate;
    }

    public int getSizeExperienceReplay() {
        return sizeExperienceReplay;
    }

    public void setSizeExperienceReplay(int sizeExperienceReplay) {
        this.sizeExperienceReplay = sizeExperienceReplay;
    }

    public Double getGamma() {
        return gamma;
    }

    public void setGamma(Double gamma) {
        this.gamma = gamma;
    }
    public Double getNoisyGreedyStd() {
        return noisyGreedyStd;
    }

    public void setNoisyGreedyStd(Double noisyGreedyStd) {
        this.noisyGreedyStd = noisyGreedyStd;
    }

    public Double getNoisyGreedyMean() {
        return noisyGreedyMean;
    }

    public void setNoisyGreedMean(Double noisyGreedyMean) {
        this.noisyGreedyMean = noisyGreedyMean;
    }

    public int getNumCriticHiddenNodes() {
        return numCriticHiddenNodes;
    }

    public void setNumCriticHiddenNodes(int numCriticHiddenNodes) {
        this.numCriticHiddenNodes = numCriticHiddenNodes;
    }

    public int getNumCriticLayers() {
        return numCriticLayers;
    }

    public void setNumCriticLayers(int numCriticLayers) {
        this.numCriticLayers = numCriticLayers;
    }

    public Double getLearning_rateCritic() {
        return learning_rateCritic;
    }

    public void setLearning_rateCritic(Double learning_rateCritic) {
        this.learning_rateCritic = learning_rateCritic;
    }

}
