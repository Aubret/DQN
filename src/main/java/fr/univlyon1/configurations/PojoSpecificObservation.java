package fr.univlyon1.configurations;

import fr.univlyon1.environment.interactions.MiniObs;
import lombok.Getter;
import lombok.Setter;

import javax.xml.bind.annotation.*;


@XmlAccessorType(XmlAccessType.FIELD)
@Setter
@Getter
public class PojoSpecificObservation<A>{

    @XmlElement(name="time")
    private Double time ;
    @XmlElement(name="id")
    private long id ;
    @XmlElement(name="alreadySent")
    private boolean alreadySent;
    @XmlElement(name="labels")
    @XmlList
    private double[] labels ;

    public PojoSpecificObservation(){}

    public PojoSpecificObservation(MiniObs miniObs){
        this.time = miniObs.getOrderedNumber();
        this.labels = miniObs.getLabels().data().asDouble();
        this.id = miniObs.getId();
        this.alreadySent = miniObs.hasAlreadySent() ;
    }
}
