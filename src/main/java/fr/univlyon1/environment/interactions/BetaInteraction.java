package fr.univlyon1.environment.interactions;

import org.nd4j.linalg.api.ndarray.INDArray;

public class BetaInteraction<A> extends Interaction<A> {
    public BetaInteraction(A action, INDArray observation, double gamma) {
        super(action, observation,gamma);
    }

    public double computeReward() {
        //System.out.println(this.gamma+" "+(this.reward*this.timefactor));
        //ouble reward = sigmo.value(3*this.secondObservation.getDouble(6));
        //double reward = this.reward * Math.max(0,Math.min(1,(this.secondObservation.getDouble(6)+1)/2));
        //System.out.println(this.secondObservation.getDouble(6)+" -> "+dt);
        double reward = ((1 - Math.exp(-this.dt*this.gamma))/this.gamma)*this.reward;
        return reward;
    }

    public double computeGamma() {
        //this.timefactor = sigmo.value(3*this.secondObservation.getDouble(6));
        //double gam =this.gamma+(1-this.gamma)*(1- Math.max(0,Math.min(1,(this.secondObservation.getDouble(6)+1)/2)));
        double gam = Math.exp(-this.dt*this.gamma);
        return gam ;
    }


    public Interaction<A> clone(){
        BetaInteraction<A> i = new BetaInteraction<A>(this.getAction(),this.getObservation(),this.gamma);
        i.setSecondObservation(this.getSecondObservation());
        i.setReward(this.getReward());
        i.setId(this.getId());
        return i ;
    }
}
