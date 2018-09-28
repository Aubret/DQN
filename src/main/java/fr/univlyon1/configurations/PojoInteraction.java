package fr.univlyon1.configurations;

import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.environment.space.ActionSpace;
import lombok.Getter;
import lombok.Setter;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.xml.bind.annotation.*;

@XmlAccessorType(XmlAccessType.FIELD)
@Setter
@Getter
public class PojoInteraction<A> implements PojoReplayable<A>{


    @XmlElement(name="observation")
    @XmlList
    private double[] observation;
    @XmlElement(name="secondObservation")
    @XmlList
    private double[] secondObservation;
    @XmlElement(name="action")
    @XmlList
    private double[] action;
    @XmlElement(name="reward")
    private double reward;

    @XmlElement(name="dt")
    private double dt;

    @XmlElement(name="time")
    private double time;

    public PojoInteraction(){}

    public PojoInteraction(Interaction interaction, ActionSpace<A> as){
        this.observation = interaction.getObservation().data().asDouble();
        this.secondObservation = interaction.getSecondObservation().data().asDouble();
        this.action = ((INDArray)as.mapActionToNumber((A)interaction.getAction())).data().asDouble();
        this.reward = interaction.getReward();
        this.time = interaction.getTime();
        this.dt = interaction.getDt();
    }

}
