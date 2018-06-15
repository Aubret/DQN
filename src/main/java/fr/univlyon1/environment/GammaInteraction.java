package fr.univlyon1.environment;

import org.apache.commons.math3.analysis.function.Sigmoid;
import org.nd4j.linalg.api.ndarray.INDArray;

public class GammaInteraction<A> extends Interaction<A> {
    public static Sigmoid sigmo = new Sigmoid();
;
    public double gamma ;
    public double timefactor ;


    public GammaInteraction(A action, INDArray observation, double gamma) {
        super(action, observation);
        this.timefactor = sigmo.value(3*observation.getDouble(6));
        this.gamma = gamma*(1-timefactor) ;

    }

    public Interaction<A> clone(){
        GammaInteraction<A> i = new GammaInteraction<A>(this.getAction(),this.getObservation(),this.gamma);
        i.setSecondObservation(this.getSecondObservation());
        i.setReward(this.getReward());
        i.setId(this.getId());
        return i ;
    }

    public double getGamma() {
        return gamma;
    }

    public void setGamma(double gamma) {
        this.gamma = gamma;
    }

    public double computeReward() {
        return this.reward*this.timefactor;
    }

}
