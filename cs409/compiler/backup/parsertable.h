//   File:          parsertable.h
//   Author(s):     R. Lancaster
//
//   Contents:
//      Declaration of the CONO table

FnPtr Parser::CONO[15][15] = {
//   + - * /         ( [         ) ]           ;      < = <=           , 
//        ->       break        call        else        end*    if while
//       int    proc program read write
//
// + or - or * or /
   {&Parser::AR,&Parser::SU,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::AR,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx},
// ( or [
   {&Parser::xx,&Parser::SU,&Parser::xx,&Parser::xx,&Parser::LD,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx},
// ) or ]
   {&Parser::LD,&Parser::SU,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::LD,&Parser::NO,&Parser::NO,&Parser::xx,&Parser::xx,&Parser::NO,
    &Parser::xx,&Parser::xx,&Parser::NO},
// ;
   {&Parser::LD,&Parser::SU,&Parser::xx,&Parser::NO,&Parser::xx,&Parser::xx,
    &Parser::LD,&Parser::NO,&Parser::NO,&Parser::EB,&Parser::EB,&Parser::NO,
    &Parser::xx,&Parser::NO,&Parser::NO},
// < or = or <=
   {&Parser::xx,&Parser::SU,&Parser::CM,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx},
// ,
   {&Parser::xx,&Parser::SU,&Parser::xx,&Parser::IO,&Parser::xx,&Parser::IO,
    &Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx},
// ->
   {&Parser::ST,&Parser::SU,&Parser::xx,&Parser::ST,&Parser::xx,&Parser::xx,
    &Parser::ST,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx},
// break
   {&Parser::xx,&Parser::xx,&Parser::xx,&Parser::BR,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx},
// call
   {&Parser::xx,&Parser::xx,&Parser::xx,&Parser::CA,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx},
// else
   {&Parser::LD,&Parser::SU,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::LD,&Parser::NO,&Parser::NO,&Parser::xx,&Parser::xx,&Parser::NO,
    &Parser::xx,&Parser::xx,&Parser::NO},
// end...
   {&Parser::xx,&Parser::xx,&Parser::xx,&Parser::NO,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx},
// if or while
   {&Parser::xx,&Parser::CN,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx},
// int
   {&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx},
// proc or program
   {&Parser::xx,&Parser::xx,&Parser::xx,&Parser::PR,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx},
// read or write
   {&Parser::xx,&Parser::SU,&Parser::xx,&Parser::IO,&Parser::xx,&Parser::IO,
    &Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,&Parser::xx,
    &Parser::xx,&Parser::xx,&Parser::xx}
//
//   + - * /         ( [         ) ]           ;      < = <=           , 
//        ->       break        call        else        end*    if while
//       int    proc program read write
};
