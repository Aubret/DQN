package fr.univlyon1.configurations;

import lombok.Getter;
import lombok.Setter;

import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlElementWrapper;
import javax.xml.bind.annotation.XmlList;
import javax.xml.bind.annotation.XmlRootElement;
import java.util.ArrayList;

@XmlRootElement(name="root")
@Getter
public class ListPojo<A> {
    @XmlElement(name="interaction")
    @XmlElementWrapper(name="interactions")
    ArrayList<PojoInteraction<A>> pojos;

    public ListPojo(){
        this.pojos = new ArrayList<>();
    }

    public void add(PojoInteraction<A> pojo){
        this.pojos.add(pojo);
    }

    public int size(){
        return this.pojos.size() ;
    }
}
