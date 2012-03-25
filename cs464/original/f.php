<?php
  
function myfunction($num, $msg) 
  {
    for($i = 0; $i != $num; $i++)
        echo $msg ."<P>";
  }

  echo "This is before the function is called<P>";
  myfunction(5,"This is a function with parameters");
  
  echo "This is after the function has been called";
  echo "<P>";
?>
