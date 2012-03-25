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
<title>Display Survey</title>
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

 $sid=$_SESSION['sid'];
// $sid=$_GET['edit_no'];
 ?>
<?
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
<form name=dis>

<table width="100%" border="0" cellspacing="0" cellpadding="0">
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
         <p><font size=5><?php echo $ti ?></font></p>
        <p><font size=4> <?php echo $de ?></font></p>

   <table width="700" align="center" height="325" border="1">
            <tr>
              <td vAlign=top><table width="615" height="86" border="0" align=left>

    <?php
$query2 =("SELECT * FROM survey_questions WHERE surid='$sid' ORDER BY ordno");
$result2=mysql_query($query2,$dbh);

if($result2)
 {
 $i=1;
  while ($details2 = mysql_fetch_object($result2))
   {
    $qno= $details2 ->qno;
    $que=$details2 ->question;
    $type=$details2 ->type;
    $surid=$details2 ->surid;
  ?>

                <tr>
                  <td><div align="left">
         <?php
        echo "<input type =checkbox name=\"Check[]\" 
value=\"$qno\"></input>\n";
          echo $i;
          echo (")"); 
          echo ($que);
           ?>
      </div>     
      
    </td>                </tr>

                <tr>
                  <td>
 <div align="left">
  
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
    
   if(is_null($choice) )
   {
  
?>
&nbsp;&nbsp;
   <input type=text name="<?php echo $chno ?>">
  <br>
<?
}
 elseif($type=="multin")
 {
?>
&nbsp;&nbsp;
  <input type =checkbox name="<? echo $chno ?>"
value="<? echo $choice ?>"> <?php echo $choice ?>
  <br>
  

<?
} 
else
{
 
?>         

&nbsp;&nbsp;  <input type="radio" name="<?php echo $qno ?>" 
 value="<?php echo $choice ?>">  <?php echo $choice ?>  
   
                       
   <?php 
} //else
}// 2 while
?>
</div><br></td>
                </tr>

<?php
} //2 if
$i++; 
} // 1while
} // if
?>


              </table>                <p>&nbsp;</p>
                <p>&nbsp;</p>                </td>
            </tr>
            </table>            
          
<tr><td></td></tr>
<tr>
<td vAlign=bottom align=center>
    </td> </tr>
<tr>
<td vAlign=bottom align=center><input type=submit value="   Add   " name=add>
   
  <input type=submit value="   Delete   " name=del>
  <input type=submit value="     Edit     " name=edit>
  <input type=submit value="Move Up" name=up>
  <input type=submit value="Move Down" name=down>
 </td></tr>
<tr><td vAlign=bottom align=center> <input type=submit value="  POST  " name=post>
</td></tr>


  </td>
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
<?
if ($_GET['post'] == "  POST  ")
 {
  //$sid=$_GET['sid'];
  //$_SESSION['lau_sid']=$sid; 
echo "<meta HTTP-EQUIV=REFRESH content=\"0;
url=http://voyager.cs.bgsu.edu/pdevu/proj/posted.php?launch=".$sid."\">";
} //if


if ($_GET['add'] == "   Add   ")
 {
 $_SESSION['sid']=$sid;
echo "<meta HTTP-EQUIV=REFRESH content=\"0;
url=survey2.php?sid=".$sid."\">";

 } //if


if ($_GET['del'] == "   Delete   ") 
 {
     foreach($Check as $no) {
  $result= mysql_query("DELETE FROM survey_questions WHERE
qno='$no'");
  $query2=("DELETE FROM question_choices WHERE
  qno=\"$qno\"");
 $result2=  mysql_query($query2,$dbh);

    }
echo "<meta HTTP-EQUIV=REFRESH content=\"0; url=displays.php\">";
   
 } //if 


if ($_GET['up'] == "Move Up")
 {
     foreach($Check as $no)
     {
     $edit_no = $no;
     } //for
   echo "<meta HTTP-EQUIV=REFRESH content=\"0;
url=http://voyager.cs.bgsu.edu/pdevu/proj/up.php?edit_no=".$edit_no."\">";

 } //if

if ($_GET['down'] == "Move Down")
 {
     foreach($Check as $no)
     {
     $edit_no = $no;
     } //for
   echo "<meta HTTP-EQUIV=REFRESH content=\"0;
url=http://voyager.cs.bgsu.edu/pdevu/proj/down.php?edit_no=".$edit_no."\">";

 } //if

if ($_GET['edit'] == "     Edit     ") 
 {
    foreach ($Check as $no)
     {  
	$edit_no = $no;
     }
	echo "<meta HTTP-EQUIV=REFRESH content=\"0;
url=http://voyager.cs.bgsu.edu/pdevu/proj/q_edit.php?edit_no=".$edit_no."\">";
 } //if 

 
 mysql_close();
 ?>
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

