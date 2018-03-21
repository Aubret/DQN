package fr.univlyon1.environment;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Interaction <A> implements Cloneable{
    private static int count = 0 ;
    private INDArray observation ;
    private INDArray secondObservation ;
    private INDArray results ;
    private A action ;
    private A secondAction ;
    private double reward ;
    private int id ;

    public Interaction(A action, INDArray observation,INDArray results){
        this.action = action ;
        this.observation = observation ;
        this.id = count ;
        this.results= results ;
        count++;
    }


    public A getAction() {
        return action;
    }

    public void setAction(A action) {
        this.action = action;
    }

    public INDArray getObservation() {
        return observation;
    }

    public void setObservation(INDArray observation) {
        this.observation = observation;
    }

    public double getReward() {
        return reward;
    }

    public void setReward(double reward) {
        this.reward = reward;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public INDArray getSecondObservation() {
        return secondObservation;
    }

    public void setSecondObservation(INDArray secondObservation) {
        this.secondObservation = secondObservation;
    }

    public INDArray getResults() {
        return results;
    }

    public void setResults(INDArray results) {
        this.results = results;
    }

    public A getSecondAction() {
        return secondAction;
    }

    public void setSecondAction(A secondAction) {
        this.secondAction = secondAction;
    }

    public Interaction<A> clone(){
        Interaction<A> i = new Interaction<A>(this.getAction(),this.getObservation(),this.getResults());
        i.setSecondObservation(this.getSecondObservation());
        i.setReward(this.getReward());
        i.setId(this.getId());
        return i ;
    }
}
