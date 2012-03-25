<?php
session_start();
?>

<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN"
"http://www.w3.org/TR/html4/loose.dtd">
<html>
<head>
<title>http://voyager.cs.bgsu.edu/pdevu/proj/survey1.htm</title>
<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">
<style type="text/css">
<!--
.style1 {
        font-size: 36px;
        font-family: Algerian;
}
.style3 {
        font-family: Verdana, Arial, Helvetica, sans-serif;
        font-size: 18px;
        font-weight: bold;
        font-variant: small-caps;
        text-transform: capitalize;
}
.style4 {font-size: 36px}
.style5 {
        font-size: 24px;
        font-weight: bold;
}
-->
</style>
</head>

<?php
$dbusername = "pdevu";
$dbpassword = "daddu";
$dbhostname = "voyager.cs.bgsu.edu";
$dbh = mysql_connect($dbhostname, $dbusername, $dbpassword)
        or die("Unable to connect to MySQL");
$selected = mysql_select_db("survey",$dbh)
        or die("Could not select Survey");
$edit_no=$_GET['edit_no'];
$query =("SELECT *  FROM survey_title WHERE id='$edit_no'");
$result=mysql_query($query,$dbh);

if($result)
 {
 while ($details = mysql_fetch_object($result))
   {
    $ti= $details ->title;
    $des =$details ->des;
   }
 } //if
?>
<body>
<table width="100%" height="419" border="1">
  <tr>
    <td align="center" valign="bottom" bgcolor="#FFCC99"><p
class="style1">&nbsp;</p>
      <p class="style1 style3 style4">ONLINE SURVEY TOOL </p>
      <p>&nbsp;</p>
      <p>&nbsp;</p>
      <p>&nbsp;</p>
  <form name="form13" method="post" action="results.php">
    <input name =sid type="hidden" value=<? echo $edit_no ?> >

      <table width="594" height="92" border="0">
       <tr> <td>
        <? echo $ti ?>
          </td>
        </tr>
       <tr><td>
         <? echo $des ?>
       </td></tr>
         <tr><td>
           <select name="restype">
                 <option value=""><I><font type= Arial></font></I></option>
                <option value="per">Percentage</option>
                <option value="charts">Charts</option>
             </select>
          </td></tr>
      </table>      <p>&nbsp;</p>
    <blockquote>

  <table width="200">
      <tr>
        <td>
         <br><br><br>
            <div align="center">
              <input type="submit" name="Submit" value="SHOW RESULTS">
            </div>
       </td>
      </tr>
    </table>
    </form>
        <p>&nbsp;</p>
    </blockquote>
    <p>&nbsp;
    </p>
    <p align="left">&nbsp;    </p>
    <p align="center"><strong> </strong></p>    <p>&nbsp;</p>
    <p>&nbsp;</p>
    <p>&nbsp;</p>
    <p>&nbsp;</p>
     <p>&nbsp;</p>
    <p>&nbsp;</p>
    <p>&nbsp;</p>

    </td>
  </tr>
</form>
</table>

</body>
</html>

