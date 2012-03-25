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

?>

<html>
<head>
<title>chart results</title>
<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">
<style fprolloverstyle>
<!--
A:hover {color: #224466}
A:link {text-decoration: none;}
A:visited {text-decoration: none;}
-->
</style>
</head>

<body bgcolor="#FFFFFF" text="#000000" marginheight=0 marginwidth=0 topmargin="0" leftmargin="0"  link="#9CBDDE" alink="#9CBDDE" vlink="#9CBDDE">
<!--3ec389ddbe44f5f8595c9172e86f90f5-->
<?php
session_start();
$dbusername = "pdevu";
$dbpassword = "daddu";
$dbhostname = "voyager.cs.bgsu.edu";
$dbh = mysql_connect($dbhostname, $dbusername, $dbpassword)
        or die("Unable to connect to MySQL");
$selected = mysql_select_db("survey",$dbh)
        or die("Could not select Survey");
$sid=$_SESSION['sid'];

 $query1 =("SELECT * FROM survey_title WHERE id='$sid'");
$result1=mysql_query($query1,$dbh);
  if($result1)
   {
  if ($details1 = mysql_fetch_object($result1))
   {
    $ti= $details1 ->title;
    $de=$details1 ->des;
  }
 }

?>
<form name=save_res1>


<table width="100%" 1="0" cellspacing="0" cellpadding="0">
  <tr>
    <td width="150"><a href="auth_accpt.php"><img src="img120.JPG" width="200" height="100"></a></td>
    <td valign="top">
      <table width="100%" border="0" cellspacing="0" cellpadding="0">
        <tr>
          <td height="69" bgcolor="#9CBDDE">
            <table width="100%" border="0" cellspacing="0" cellpadding="0">
              <tr>
                <td><font size=6><center>ONLINE SURVEY TOOL</center></font></td>
              </tr>
            </table>
          </td>
        </tr>
        <tr>
          <td> &nbsp;</td>
        </tr>
      </table>
    </td>
  </tr>
  <tr>
    <td valign="top">
      <table width="175" border="0" cellspacing="0" cellpadding="0" bgcolor="#3984BD">
        <tr> 
          <td colspan="2"></td>
        </tr>
          <tr> 
          <td>&nbsp;</td>
          <td><a href="auth_accpt.php""><font face="Tahoma, Verdana, Arial" size="4">HOME
           </font></a></td>
        </tr>

        <tr> 
          <td>&nbsp;</td>
          <td><a href="http://voyager.cs.bgsu.edu/pdevu/proj/survey1.php"><font face="Tahoma, Verdana, Arial" size="4">CREATE 
            SURVEY </font></a></td>
        </tr>
        <tr> 
          <td>&nbsp;</td>
          <td><font face="Tahoma, Verdana, Arial" size="4"><a href="http://voyager.cs.bgsu.edu/pdevu/proj/td.php">MY  
            SURVEYS</a></font></td>
        </tr>
        <tr> 
          <td>&nbsp;</td>
          <td><a href="http://voyager.cs.bgsu.edu/pdevu/proj/logout.php"><font size="4" face="Tahoma, Verdana, Arial">LOGOUT
            </font></a></td>
        </tr>
        
        <tr> 
          <td colspan="2"></td>
        </tr>
      </table>
      </td>
    <td valign="top">
      <table width="95%" height=700 border="0" cellspacing="0" cellpadding="0">
        <tr>
          <td vAlign=top><div align=center>
          <p><font size=5><?php echo "RESULTS FROM ONLINE SURVEY TOOL" ?></font></p>
         <p><font size=5><?php echo $ti ?></font></p>
        <p><font size=4> <?php echo $de ?></font></p>

   <table width="700" align="center" height="50" border="0">
            <tr>
              <td vAlign=top>
                 <table width="615" height="10" border="1" align=left>

    <?php
     function graphic($per)
     {
      $pervalue=($per * 200);
       echo "
       <img src=\"gd.php?per=$per\" height=\"50\"</td>$per%";
     }

 $qnumber=array();
$count=0;
$total=0;

$query2 =("SELECT * FROM survey_questions WHERE surid='$sid' ORDER BY
qno");

$t=mysql_num_rows($query2,$dbh);
echo $t;

$result2=mysql_query($query2,$dbh);

if($result2)
 {
 $i=1;
  while ($details2 = mysql_fetch_object($result2))
   {
    $qno= $details2 ->qno;
    $qnumber[]=$qno;
    $que=$details2 ->question;
    $type=$details2 ->type;
    $surid=$details2 ->surid;
  ?>

                <tr>
                  <td vAlign="top"><div align="left">
         <?php
          echo $t;
          echo $i;
          echo (")"); 
          echo ($que);
           ?>
           
   <table width=100 border=0>
           <?php
  $query3 = ("SELECT chno, choice FROM question_choices WHERE
qno='$qno' ORDER BY chno");
  $result3=mysql_query($query3,$dbh);

if($result3)
 {
  while ($details3 = mysql_fetch_object($result3))
   {
    $chno= $details3 ->chno;
    $choice=$details3 ->choice;
    
   if(!is_null($choice) )
   {
  $query5=("SELECT qno, ans, COUNT(*) as count FROM answers where
ans='$choice' AND qno='$qno' GROUP BY qno,ans");
  $result5=mysql_query($query5,$dbh);
    if($result5)
     {
     while ($details5 = mysql_fetch_object($result5))
     $count=$details5 ->count;
    }
$query6=("SELECT qno,COUNT(*) as total FROM answers WHERE
qno='$qno' GROUP BY qno");
  $result6=mysql_query($query6,$dbh);
    if($result6)
     {
     while ($details6 = mysql_fetch_object($result6))
     $total=$details6 ->total;
    }

?>
  
  <tr>
<td vAlign="top">&nbsp;&nbsp;<div align="left">
<?php echo $choice ?>

</div></td>
<td vAlign="top"> 
<?
    $per=round(($count/$total) * 100,2);
    $count=0;
    $total=0;
    graphic($per);
?>
 </tr>

  <?php
  } // if

         elseif(is_null($choice) && $type=="st" )

         {
   ?>  
   <tr><td vAlign="top"> 

        <select name="<? echo $chno ?>" >
    <option value="results"><bgcolor="#ff6666"><? echo
"Results..." ?></option>
<?
         $query9=("SELECT ans FROM answers WHERE qno='$qno'");
   $result9=mysql_query($query9,$dbh);
      while ($details9 = mysql_fetch_object($result9))
      {
    $ans= $details9 ->ans;
     ?>  
   <option value="<? echo $ans; ?>"> <? echo $ans; ?></option>
    <?
      } //while
    ?>
</select>

   </td></tr>
   
<?
} // else
  } // 2 while choices
 } // if choices

 if($type=="mt" )

         {
      ?>  
   <tr><td vAlign="top"> 

        <select name="<? echo $qno ?>" >
    <option value="results"><bgcolor="#ff6666"><? echo "Results..." ?></option>
       <?
        $query=("SELECT DISTINCT(taker_id) FROM answers WHERE sid='$sid'");
        $result=mysql_query($query,$dbh);

       while ($details = mysql_fetch_object($result)) 
        {
         $res=$details ->taker_id;

         $query14=("SELECT ans FROM answers WHERE qno='$qno' and taker_id='$res' ORDER BY id DESC");
   $result14=mysql_query($query14,$dbh);
      while ($details14 = mysql_fetch_object($result14))
      {
    $ans= $details14 ->ans;
     ?>  
   <option value="<? echo $ans; ?>"> <? echo $ans; ?></option>

    <?
      } //while
    ?>
 
<option value="separate"> <? echo "-----------------------------"; ?></option>

<?
  } // while taker
?>
</select>

   </td></tr>
   
<?
} // if mt

 ?>

</table>
</div></td></tr>
<?php
$i++; 
} // 1while
} // if
 mysql_close();
?>


              </table>                <p>&nbsp;</p>
                <p>&nbsp;</p>         
            
             </td>
            </tr>
            </table>            
      <tr><td><br></td></tr>    
     <tr>
       <td> <div align=center>
       <a href="percentages.php?edit_no=<?php echo $sid; ?>">PERCENTAGES</a> &nbsp;&nbsp;&nbsp;&nbsp;

       <a href="spreadsheet.php?edit_no=<?php echo $sid; ?>">DOWNLOAD TO SPREADSHEET</a>

        
      </div></td>
  
  
     </tr>
        <tr>
          <td height="50" valign="bottom"> 
            <div align="center"><font size="1" face="Arial, Helvetica, sans-serif">&copy; 
              Copyright [Purnima Devu]. All Rights Reserved.</font></div>
          </td>
        </tr>
      </table>
    </td>
  </tr>
</table>
</form>
</body>
</html>

<?
} //if
else
{
?>
<META HTTP-EQUIV="refresh" content="0;url=http://voyager.cs.bgsu.edu/pdevu/proj/first_login.php">
<?
} //else
?>
