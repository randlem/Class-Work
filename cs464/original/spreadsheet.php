<?
session_start();
$u=$_SESSION['uid'];

$dbusername = "pdevu";
$dbpassword = "daddu";
$dbhostname = "voyager.cs.bgsu.edu";	
$dbh = mysql_connect($dbhostname, $dbusername, $dbpassword) 
	or die("Unable to connect to MySQL");
$selected = mysql_select_db("survey",$dbh) 
	or die("Could not select survey");


$query = ("SELECT username FROM login WHERE usid='$u'");
$result = mysql_query($query,$dbh);


if($result)
 {
 if ($details1 = mysql_fetch_object($result))
    {
     $usname= $details1 ->username;
    }
}

 if(!is_null($usname))
 {

 $launch_surid=$_GET['edit_no'];
header("Content-Type: application/vnd.ms-excel"); 
header("Content-Disposition: inline; filename=\"<? echo $launch_surid ?>.xls\"");  
?> 


<?php

$query2 =("SELECT * FROM survey_questions WHERE surid='$launch_surid'
ORDER BY qno");
$result2=mysql_query($query2,$dbh);
//unset($query2);
if($result2) // if 1
 {
 $i=1;
  while ($details2 = mysql_fetch_object($result2)) // while 1
   {
    $qno= $details2 ->qno;
    $ty= $details2 ->type;
    $que=$details2 ->question;
    $surid=$details2 ->surid;

   if($ty=="mt" || $ty=="multin")
   {
  
    $query3 = ("SELECT chno, choice FROM question_choices WHERE
qno='$qno' ORDER BY chno");
  $result3=mysql_query($query3,$dbh);
$num_rows = mysql_num_rows($result3);

  for($k=1;$k<=$num_rows;$k++)
   {     
    echo "Q".$i."[".$k."]"."\t";

   }
    $i=$i+1;

   }

   else
   {
  echo "Q".$i."\t";
  $i=$i+1;
   }

 } //w

echo "\n";
} // if



 $query=("SELECT DISTINCT(taker_id) FROM answers WHERE sid='$launch_surid'");
   $result=mysql_query($query,$dbh);

    if($result)
     {
      while ($details = mysql_fetch_object($result)) 
      {
         $res=$details ->taker_id;

       $query4 =("SELECT * FROM survey_questions WHERE surid='$launch_surid' ORDER BY qno");
       $result4=mysql_query($query4,$dbh);

    if($result4) // if 1
   while ($details4 = mysql_fetch_object($result4)) // while 1
   {
    $qno= $details4 ->qno;
    $ty= $details4 ->type;
      
      
  if($ty=="multin")
         {
  
    $query3 = ("SELECT chno, choice FROM question_choices WHERE qno='$qno' ORDER BY chno");
    $result3=mysql_query($query3,$dbh);
    $num_rows = mysql_num_rows($result3);
         

   $query5=("SELECT ans FROM answers where taker_id='$res' and qno='$qno' ORDER by id DESC");
   $result5=mysql_query($query5,$dbh);
   $num_ans=mysql_num_rows($result5);

        if($result5)
        {
     while ($details5 = mysql_fetch_object($result5))
       {
        $ans=$details5 ->ans;

        echo $ans."\t";
      } //w
    
     $tabs=$num_rows-$num_ans;
  for($r=1;$r<=$tabs;$r++)
    echo "\t";

    } // if 
  } //mt,multin


    else
    {
    $query5=("SELECT ans FROM answers where taker_id='$res' and qno='$qno' ORDER by id DESC");
   $result5=mysql_query($query5,$dbh);
   $num_ans=mysql_num_rows($result5);

        if($result5)
        {
     while ($details5 = mysql_fetch_object($result5))
       {
         $ans=$details5 ->ans;

        echo $ans."\t";
      } //w
    } // if    

    } // else
  } //while q

    echo "\n";  
} //w
} // if
mysql_close();

} //if
else
{
?>
<META HTTP-EQUIV="refresh" content="0;url=http://voyager.cs.bgsu.edu/pdevu/proj/first_login.php">
<?
} //else
?>


  
