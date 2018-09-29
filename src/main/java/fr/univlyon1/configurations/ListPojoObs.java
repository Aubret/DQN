package fr.univlyon1.configurations;

import lombok.Getter;

import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlElementWrapper;
import javax.xml.bind.annotation.XmlRootElement;
import java.util.ArrayList;

@XmlRootElement(name="root")
@Getter
public class ListPojoObs<A> {
    @XmlElement(name="spo")
    @XmlElementWrapper(name="spoList")
    ArrayList<PojoSpecificObservation<A>> pojos;

    public ListPojoObs(){
        this.pojos = new ArrayList<>();
    }

    public void add(PojoSpecificObservation<A> pojo){
        this.pojos.add(pojo);
    }

    public int size(){
        return this.pojos.size() ;
    }
}
