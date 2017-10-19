import java.util.ArrayList;
import java.util.List;

public class TreeNode {
	
  private Object splitValue;
  private Integer splitIndex;
  private List<TreeNode> children;
  private TreeNode parent;
  private Integer positiveInstances;
  private Integer negativeInstances;
  private String operator;
  private Integer positiveCount;
  private Integer negativeCount;
  
  public TreeNode()
  {
	 this.children = new ArrayList<TreeNode>();  
  }
  
  public Integer getSplitIndex() {
	return splitIndex;
  }
  public void setSplitIndex(Integer splitIndex) {
	this.splitIndex = splitIndex;
  }
  public Object getSplitValue() {
	return splitValue;
  }
  public void setSplitValue(Object splitValue) {
	this.splitValue = splitValue;
  }
  public List<TreeNode> getChildren() {
	return children;
  }
  public void setChildren(List<TreeNode> children) {
	this.children = children;
  }

  public TreeNode getParent() {
	return parent;
  }

  public void setParent(TreeNode parent) {
	this.parent = parent;
  }

  public Integer getPositiveInstances() {
	return positiveInstances;
  }

   public void setPositiveInstances(Integer positiveInstances) {
	this.positiveInstances = positiveInstances;
  }

  public Integer getNegativeInstances() {
	return negativeInstances;
  }

   public void setNegativeInstances(Integer negativeInstances) {
	this.negativeInstances = negativeInstances;
  }

  public String getOperator() {
	return operator;
  }

  public void setOperator(String operator) {
	this.operator = operator;
  }

  public Integer getPositiveCount() {
	return positiveCount;
  }

  public void setPositiveCount(Integer positiveCount) {
	this.positiveCount = positiveCount;
  }

  public Integer getNegativeCount() {
	return negativeCount;
  }

  public void setNegativeCount(Integer negativeCount) {
	this.negativeCount = negativeCount;
  }
   
}
