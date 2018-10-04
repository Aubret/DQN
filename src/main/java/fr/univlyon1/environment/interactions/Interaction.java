package fr.univlyon1.environment.interactions;

import fr.univlyon1.environment.space.SpecificObservation;
import lombok.Getter;
import lombok.Setter;
import org.nd4j.linalg.api.ndarray.INDArray;

@Setter
@Getter
public class Interaction <A> implements Replayable<A>{
    protected static int count = 0 ;
    // time of first observation
    protected Double time ;
    // time between the first and second observation
    protected Double dt ;
    protected INDArray observation ;
    protected INDArray secondObservation ;

    protected A action ;
    protected A secondAction ;
    protected double reward ;
    protected double gamma ;
    protected int id ;
    protected long idObserver ;


    public Interaction(A action, INDArray observation){
        this.action = action ;
        this.observation = observation ;
        this.id = count ;
        count++;
    }

    public Interaction(A action, INDArray observation,double gamma){
        this.action = action ;
        this.observation = observation ;
        this.id = count ;
        this.gamma = gamma ;
        count++;
    }

    public Interaction<A> clone(){
        //A action = (A)((ContinuousAction)this.getAction()).copy();
        Interaction<A> i = new Interaction<A>(this.getAction(),this.getObservation(),this.gamma);
        i.setSecondObservation(this.getSecondObservation());
        i.setReward(this.getReward());
        i.setId(this.getId());
        i.setIdObserver(this.getIdObserver());
        i.setTime(this.time);
        i.setDt(this.dt);
        return i ;
    }

    public double computeReward() {
        return this.reward;
    }
    public double computeGamma() {
        return this.gamma;
    }
    public SpecificObservation emitObs(){
        return new MiniObs(this);
    }
}
