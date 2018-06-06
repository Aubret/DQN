package fr.univlyon1.memory.sumTree;

import fr.univlyon1.memory.prioritizedExperienceReplay.InteractionHistory;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class Node<A>{
    private Node<A> left;
    private Node<A> right ;
    private Double sum ;
    private InteractionHistory<A> ih ;

    private Node<A> min ;
    private Node<A> max ;

    private Node<A> parent;
    private int numberDescendant ;

    public Node(InteractionHistory<A> ih,Node<A> parent){
        this.left = null ;
        this.right = null ;
        this.min = this ;
        this.max = this ;
        if(ih != null) {
            this.ih = ih;
            this.sum = ih.getErrorValue();
        }
        this.parent = parent;
        this.numberDescendant = 0 ;
    }

    public Node<A> getUp(Double value,int id){

        if(ih.getErrorValue().equals(value) && ih.getId() == id) {
            return this.reajust(); // Cas où on le trouve
        }
        Node<A> retour = null ;
        if(value <= ih.getErrorValue()  && this.left != null){
            retour = this.left.getUp(value,id); // Il est à gauche
        }

        if(value >= ih.getErrorValue() && retour == null && this.right != null){
            retour = this.right.getUp(value,id); //Avec les repositionnement, peut-être que ce cas est possible...
        }
        this.majSum();
        return retour ;
    }

    public Node<A> getUp(Double value){
        if(this.left != null){
            double val = value - this.left.getSum() ;
            if(val <= 0) { // Le candidat se situe à gauche de l'arbre
                Node<A> n = this.left.getUp(value);
                this.majSum();
                return n;
            }else
                value = val ;
        }
        double val = value - this.ih.getErrorValue() ;
        if(val <= 0) { // On a trouvé le bon candidat, il faut maintenant réajuster l'arbre
            //System.out.println("here");
            return this.reajust();
        }
        value = val ;
        if( this.right == null || value > this.right.getSum()){
            System.out.println("erreur valeur");
            System.out.println("parent "+this.parent.getSum()+" "+this.parent.getIh().getErrorValue());
            if(this.parent.left != null)
                System.out.println("parent left"+this.parent.left.getSum()+" "+this.parent.left.getIh().getErrorValue());
            if(this.parent.right != null)
                System.out.println("parent right"+this.parent.right.getSum()+" "+this.parent.right.getIh().getErrorValue());
            if(this.left != null)
                System.out.println("left"+this.left.getSum()+" "+this.left.getIh().getErrorValue());
            System.out.println("me : "+this.getSum()+" "+this.ih.getErrorValue());
            System.out.println(value);
            System.out.println(this.right.getSum());
        }
        Node<A> n = this.right.getUp(value);// Le noeud se situe à droit ed el'arbre
        this.majSum();
        return n;
    }


    public void insert(InteractionHistory<A> ih) {
        if(ih.getErrorValue() <= this.ih.getErrorValue()){
            if(this.left == null){
                this.left = new Node<A>(ih,this);
                //System.out.println(this.getIh().getErrorValue()+" left :"+this.left.getIh().getErrorValue());
            }else{
                //System.out.println(this);
                this.left.insert(ih);
            }
        }else{
            if(this.right == null){
                this.right = new Node<A>(ih,this);
                //System.out.println(this.getIh().getErrorValue()+" right :"+this.right.getIh().getErrorValue());
            }else{
                this.right.insert(ih);
            }
        }
        this.majSum();
    }


    public void remove(Node<A> node,Node<A> attach){
        if(this.left == node) {
            this.left = attach;
            if (attach != null)
                attach.setParent(this);
        }else if(this.right == node) {
            this.right = attach;
            if(attach !=null)
                attach.setParent(this);
        }else{
            System.out.println("BUG PARENT");
            System.out.println(this+" "+this.left +" "+this.right+" "+node+" "+attach);
        }
    }

    public Node<A> removeFirst() {
        if(this.left ==null){
            this.parent.remove(this,this.right);
            return this ;
        }
        Node<A> n = this.left.removeFirst();
        this.majSum();
        return n ;
    }

    public Node<A> removeLast() {
        if(this.right ==null){
            this.parent.remove(this,this.left);
            return this ;
        }
        Node<A> n = this.right.removeLast();
        this.majSum();
        return n ;
    }

    protected Node<A> reajust(){
            //System.out.println("here");
        int numberL = this.left != null ? this.left.getNumberDescendant()+1 : -1 ;
        int numberR =  this.right != null ? this.right.getNumberDescendant()+1 : -1 ;
        Node<A> changePlace=null; // Il faut récupérer le noeud qiu remplacera le noeud qu'on enlève
        if(numberL > numberR){ //left != -1 et left > right
            changePlace = this.left.removeLast();
        }else if(this.right != null){ // right != -1 et right >= left
            changePlace = this.right.removeFirst() ;
        }else{
            if(this.left != null || this.right != null){
                System.out.println("LA BUG");
            }
        }

        if(changePlace != null) { // right || left != null
            if(this.left != changePlace) {
                changePlace.setLeft(this.left); // On met à jour l'arbre
                if(this.left != null)
                    this.left.setParent(changePlace);
            }else
                changePlace.setLeft(null);

            if(this.right != changePlace) {
                changePlace.setRight(this.right);
                if(this.right != null)
                    this.right.setParent(changePlace);
            }else
                changePlace.setRight(null);
            changePlace.majSum();
        }
        this.parent.remove(this,changePlace);
        return this;
    }

    public void print(){
        if(this.left != null){
            this.left.print();
        }
        System.out.println("error : "+this.getIh().getErrorValue()+"number : "+this.ih.getSumValues()+" ; sum : "+this.sum);
        if(this.right != null){
            this.right.print();
        }
    }

    public void majSum(){
        this.sum =( this.left!= null ? this.left.getSum() :0. )+ ( this.right!= null ? this.right.getSum():0. ) + this.ih.getErrorValue();
        this.numberDescendant = ( this.left!= null ?this.left.getNumberDescendant()+1 :0 )+ ( this.right!= null ? this.right.getNumberDescendant()+1:0 );
        this.min = this.left != null  ? this.left : this ;
        this.max = this.right != null ? this.right : this ;
    }

}
