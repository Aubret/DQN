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
    private double alpha ;


    public InteractionHistory(Interaction<A> interaction, double error){
        this.interaction = interaction ;
        this.errorValue = error ;
        this.sumValues = 0. ;
        this.epsilon = 0.01 ;
        this.alpha = 1.;
    }

    public void computeError(double error){
        error = error + epsilon ;
    }
}
