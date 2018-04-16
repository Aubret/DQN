package fr.univlyon1.configurations;

import fr.univlyon1.environment.Interaction;
import fr.univlyon1.environment.space.ActionSpace;
import lombok.Getter;
import lombok.Setter;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;

@XmlRootElement(name="interaction")
@XmlAccessorType(XmlAccessType.FIELD)
@Setter
@Getter
public class PojoInteraction<A> {


    public PojoInteraction(Interaction interaction, ActionSpace<A> as){
        this.observation = interaction.getObservation().data().asDouble();
        this.secondObservation = interaction.getSecondObservation().data().asDouble();
        this.action = ((INDArray)as.mapActionToNumber((A)interaction.getAction())).data().asDouble();
        this.reward = interaction.getReward();
    }

    @XmlElement(name="observation")
    private double[] observation;
    @XmlElement(name="secondObservation")
    private double[] secondObservation;
    @XmlElement(name="action")
    private double[] action;
    @XmlElement(name="reward")
    private double reward;
}
