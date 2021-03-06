package fr.univlyon1.memory.prioritizedExperienceReplay;

import fr.univlyon1.environment.interactions.Interaction;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class InteractionHistory<A> {
    private Interaction<A> interaction ;
    private int id ;
    private Double errorValue ;
    private Double sumValues ;
    private double epsilon ;

    private double errorFactor=1. ;


    public InteractionHistory(Interaction<A> interaction, double error){
        this.interaction = interaction ;
        this.errorValue = error ;
        this.epsilon = 0.05 ;
        this.sumValues=0. ;
        this.id = interaction.getId();
    }

    public void computeError(double error){
        // C'est normalementla valeur absolue de la TD error, il fautenlever lecarré
        this.sumValues++ ;
        //error =1. / (1. + Math.exp(-5*error));
        //this.errorFactor = 0.99*this.errorFactor ;
        //this.errorValue =Math.pow(error,2) ;//*(1-val);//-0.05*sumValues;
        this.errorValue = error;
    }


}
