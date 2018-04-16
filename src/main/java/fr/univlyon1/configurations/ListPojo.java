package fr.univlyon1.configurations;

import lombok.Getter;
import lombok.Setter;

import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlList;
import javax.xml.bind.annotation.XmlRootElement;
import java.util.ArrayList;

@XmlRootElement(name="list")
@Setter
@Getter
public class ListPojo<A> {
    @XmlElement(name="interactions")
    @XmlList
    private ArrayList<PojoInteraction<A>> pojos;

    public void add(PojoInteraction<A> pojo){
        this.pojos.add(pojo);
    }
}
