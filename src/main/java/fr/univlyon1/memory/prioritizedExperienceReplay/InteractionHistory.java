package fr.univlyon1.memory.prioritizedExperienceReplay;

import fr.univlyon1.environment.Interaction;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class InteractionHistory<A> {
    private Interaction<A> interaction ;
    private Double errorValue ;
    private Double sumValues ;
    private double epsilon ;


    public InteractionHistory(Interaction<A> interaction, double error){
        this.interaction = interaction ;
        this.errorValue = error ;
        this.epsilon = 0.05 ;
        this.sumValues=0. ;
    }

    public void computeError(double error){
        //this.sumValues++ ;
        //double val = Math.min(sumValues/30.,1.);
        this.errorValue = Math.pow(error,2);//*(1-val);//-0.05*sumValues;
    }


}
