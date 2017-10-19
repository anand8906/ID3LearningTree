import java.util.ArrayList;
import java.util.List;

public class NominalSplit extends Split{
	
  List<String> splitValues;
  
  public NominalSplit() {
	  super();
      splitValues = new ArrayList<>();	  
  }

  public List<String> getSplitValues() {
		return splitValues;
  }
	
  public void setSplitValues(List<String> splitValues) {
		this.splitValues = splitValues;
  }
  
}
