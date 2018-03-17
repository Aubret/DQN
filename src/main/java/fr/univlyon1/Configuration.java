package main.java.fr.univlyon1;

public class Configuration {
    int epochs = 50; // Nombre de time step avant la fusion des approximateurs
    int iterations = 1; // Nombre d'itération d'apprentissage à chaque timestep
    int batchSize = 20; // Taille des batchs pour l'apprentissage
    Double minEpsilon = 0.01; // epsilon minimum dans epsilon decrmental
    int stepEpsilon = 100; // nomre de pas pour décrémenter epsilon
    int numHiddenNodes = 10; // Nombre de couches cachées
    int numLayers = 1; // Nombre de couches
    Double learning_rate = 0.001; // Pas d'apprentissage
    int sizeExperienceReplay = 5000; // Taille du buffer de l'expérience replay
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
}
