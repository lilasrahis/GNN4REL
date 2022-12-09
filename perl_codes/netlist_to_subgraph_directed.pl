#! /bin/env perl
########Conver netlist to adj matrix
require 5.004;
use FindBin;    # New in Perl5.004
use List::Util qw/shuffle/;
require "./theCircuit.pm";
use File::Path qw( make_path );
use File::Spec;
my @functions=();
my $file_name="";
my $assign_count=0;
my $ml_count=0;
my %features_map=();
$features_map{"AOI"}=10;
$features_map{"CLKBUF"}=11;
$features_map{"CLKGATETST"}=12;
$features_map{"DFFRNQ"}=13;
$features_map{"DFFSNQ"}=14;
$features_map{"FA"}=15;
$features_map{"HA"}=16;
$features_map{"LHQ"}=17;
$features_map{"MUX"}=18;
$features_map{"OAI"}=19;
$features_map{"SDFFRNQ"}=20;
$features_map{"SDFFSNQ"}=21;
$features_map{"in_degree"}=22;
$features_map{"out_degree"}=23;
$features_map{"driving"}=24;
$features_map{"PI"}=0;
$features_map{"PO"}=1;
$features_map{"XOR"}=3;
$features_map{"XNOR"}=4;
$features_map{"AND"}=5;
$features_map{"OR"}=6;
$features_map{"NAND"}=7;
$features_map{"NOR"}=8;
$features_map{"INV"}=9;
$features_map{"BUF"}=2;
my $start_time               = time;
my ($rel_num)                = '$Revision: 1.7 $' =~ /\: ([\w\.\-]+) \$$/;
my ($rel_date) = '$Date: 2022/12/09 20:38:38 $' =~ /\: ([\d\/]+) /;
my $prog_name = $FindBin::Script;

my $hc_version = '0.1';

my $help_msg = <<HELP_MSG;
This program converts a verilog netlist into a graph
Usage: perl $prog_name  -i <input_dataset_name> -f <output_folder_name> -m <start_node_count> > log.txt

    Options:	-h | -help		 Display this info

                -v | -version	 Display version & release date

                -i               Input dataset name
                
                -f               Output folder name 

                -m               Starting count for the nodes in the graph. Default is 0

    Example:

    UNIX-SHELL> perl netlist_to_subgraph_directed.pl -i test_adder -f test_adder -m 0 > log_adder.txt


HELP_MSG

format INFO_MSG =
     @|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
     $prog_name
     @|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
     "Version $rel_num  Released on $rel_date"
     @|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
     'Lilas Alrahis <lma387@nyu.edu>'
     @|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
     'NYUAD, Abu Dhabi, UAE'

     @|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
     "\'$prog_name -help\' for help"

.

use subs
  qw(PrintWarning PrintError PrintFatalError PrintInternalError PrintDebug);

my $error            = 0;
my $input_file;
my $input_dir;
my $comment = 0;
while ( $_ = $ARGV[0], /^-/ ) {              # Get command line options
    shift;
    if (/^-h(elp)?$/) { $~ = "INFO_MSG"; write; print $help_msg; exit 0 }
elsif (/^-f(ile)?$/){$file_name_=shift;}    

    elsif (/^-m(lcount)?$/)     { $ml_count       = shift; }
elsif (/^-i(nput)?$/)     { $input_dir       = shift; }
    else                      { PrintError "Unknown option: '$_'!" }
}

if ( !( defined($input_dir) ) ) {
    PrintError "Expect an input Verilog files!";
}
if ( $error > 0 ) {
    warn "\n$help_msg";
    exit 1;
}

select( ( select(STDERR), $~ = "INFO_MSG" )[0] ), write STDERR;

###################################################
#################### Started Here
###################################################

print "Starting with node count $ml_count\n";
my $status = 0;
my $filename_cell = 'cell.txt';
my $filename_count="count.txt";
$file_name_="../Path_PNA/data/".$file_name_;
system("mkdir -p $file_name_");
open(FH_LINK, '>',"${file_name_}/link.txt") or die $!;
open(FH_CELL, '>', "${file_name_}/$filename_cell") or die $!;
open(FH_COUNT, '>', "${file_name_}/$filename_count") or die $!;
my $filename_feat = 'feat.txt';
open(FH_FEAT, '>', "${file_name_}/$filename_feat") or die $!;
opendir my $dh, $input_dir or die "Cannot open $input_dir: $!";
my @input_files = sort grep { ! -d } readdir $dh;
closedir $dh;
foreach my $input_file (@input_files) {


next if ($input_file=~m/^\./);
my %the_circuit              = ();
my @list_of_gates=();
my %Netlist_Outputs_Hash =();
my %Netlist_Inputs_Hash=();
my $driving="";    
my $cell_name           = "";
    my $instance_name       = ""; 
    my @ports               = ();
    my %ports               = ();
    my $multi_line_statment = "";
    my $connect_this_line   = 0;
    my $a                   = "";
    my $line                = "";
    my @Netlist_Inputs      = ();
    my @Netlist_Outputs     = ();
    my @Module_Inputs      = ();
    my @Module_Outputs     = ();


    local *INPUT_FH;     # Only way to declare a non-global filehandler.

    ##################Now I wanna parse again to get the list of inputs, outputs and gates
    open INPUT_FH, "${input_dir}/${input_file}"
      or PrintFatalError "Can't open input file '$input_file': $!!";
   
 while (<INPUT_FH>) {
        $line = $_;
                if ( $line =~ /^\s*(module)\s+(\w+)\b/ ) {

             
				}
			 elsif ($line =~ /^\s*(endmodule)\b/) {
@Netlist_Inputs=@Module_Inputs  ;   
@Netlist_Outputs= @Module_Outputs ;  

	   @Module_Inputs      = ();
       @Module_Outputs     = ();
			 
			 }
        elsif ( $line =~ /^\s*(input)\s+.*/ ) {   
            if ( $line =~ /^.*?\;\s*$/ ) {
                $line =~ s/input\s+//;             
                $line =~ tr/;//d; 
                for ($line) {
                    s/^\s+//;
                }
                $line =~ s/\R//g;
                my @found_inputs = split( /,/, $line ); 
                push @Module_Inputs, @found_inputs;
                next;  
            }
            else {     
                until ( $line =~ /^.*?\;\s*$/ ) {
                  $line =~ s/input\s+//g; 
                    $line =~ tr/;//d; 
                    for ($line) {
                        s/^\s+//;
                    }
                    $line =~
                      s/\R//g;  
                    my @found_inputs = split( /,/, $line );
                    push @Module_Inputs, @found_inputs;
                    $line = <INPUT_FH>;

                }                       
                $line =~ s/input\s+//g; 
                $line =~ tr/;//d;       

                for ($line) {
                    s/^\s+//;
                }
                $line =~
                  s/\R//g;   
                my @found_inputs = split( /,/, $line );
                push @Module_Inputs, @found_inputs;

                next;
            }
        }   

        if ( $line =~ /^\s*(output)\s+.*/ ) { 
            if ( $line =~ /^.*?\;\s*$/ ) {
                $line =~ s/output//;      
                $line =~ tr/;//d;        
                for ($line) {
                    s/^\s+//;
                }
                $line =~ s/\R//g;  
                my @found_outputs = split( /,/, $line );  
                push @Module_Outputs,   @found_outputs;
                next;   
            }
            else {     
                until ( $line =~ /^.*?\;\s*$/ ) {
                    $line =~ s/output//g;  
                    $line =~ tr/;//d;   
                    for ($line) {
                        s/^\s+//;
                    }
                    $line =~
                      s/\R//g; 
                    my @found_outputs = split( /,/, $line );
                    push @Module_Outputs,   @found_outputs;
                    $line = <INPUT_FH>;
                }             
                $line =~ s/output//g;
                $line =~ tr/;//d;   
                for ($line) {
                    s/^\s+//;
                }
                $line =~
                  s/\R//g;   
                my @found_outputs = split( /,/, $line );
                push @Module_Outputs,   @found_outputs;
                next;
            }
        } 

        
    }
    close INPUT_FH;

my @tempp=();
foreach my $inn (@Netlist_Inputs){
$inn=~ s/^\s+|\s+$//g;
if ($inn=~m/\s*\[(\d+)\:(\d+)\]\s+(\S+)/){
my $start=$1;
my $end=$2;
my $name=$3;
my $i=$start;
if ($start>$end){
$i=$end;
$end=$start;
}
while ($i<=$end)
{
push @tempp, "$name\[$i\]";
$i++;
}

}
else{
push @tempp, $inn;
}
}
 @Netlist_Inputs=@tempp;
   @tempp=();


 
    %Netlist_Outputs_Hash = map { $_ => 1 } @Netlist_Outputs;
     %Netlist_Inputs_Hash  = map { $_ => 1 } @Netlist_Inputs;

#######################open file again to initialize the circuit
    
	open INPUT_FH, "${input_dir}/${input_file}"
      or PrintFatalError "Can't open input file '$input_file': $!!";
    while (<INPUT_FH>) {
        @ports = ();
        %ports = ();
        $line  = $_;
	if ( $line =~ /^\s*(wire)\s+.*/ ) {    #############check inputs

        }    #### end of wire detection

if ($line=~m/^\s*assign\s+(\S*)\s+=\s+(\S*)\;/){
my $out=$1;
my $in=$2;

my $modified_name="assign_${assign_count}";

push @list_of_gates, $modified_name;
my $current_object;
my @current_gate_inputs=();
push @current_gate_inputs, $in;
my @current_gate_outputs=();
push @current_gate_outputs, $out;
                    $current_object = theCircuit->new(
                        {	
                            name          => $modified_name, 
                            bool_func     => "BUF",

			    processed => "X1",
                            inputs        => \@current_gate_inputs,
                            outputs        => \@current_gate_outputs,
                            fwdgates => [undef],
                            fwdgates_inst => [undef],
                            count =>$ml_count,
                        }
                    );
			my $indicator=0;
	 	foreach my $current_gate_output (@current_gate_outputs){
                    if ( exists( $Netlist_Outputs_Hash{$current_gate_output} ) )
                    { 
					my @temp=();
					my @temp_inst=();
					if ($indicator==0){
					
                    push @temp, "PO";
                    
                    push @temp_inst, $current_gate_output;
					}
					else{
					@temp=$current_object->get_fwdgates();
					@temp_inst=$current_object->get_fwdgates_inst();
					push @temp, "PO";
                    
                    push @temp_inst, $current_gate_output;
					}
					$indicator++;
					
					
                      $current_object->set_fwdgates(\@temp);
                      $current_object->set_fwdgates_inst(\@temp_inst);
                    }
					}
                                    $the_circuit{$modified_name} = $current_object;
$ml_count++;
$assign_count++;
}
elsif ($line  =~ m/

	^\s*
	(\S*)  # Cell name
	\s+
	(\S*)  #Instance Name
	\s*
	\(
	.+   #ports list
	$/x
       ){
	
      if ( $line =~ /\;/ )
	{
	  
	} else {
	  until ( $line =~ /\;\s*$/ )
	    {
	  
$line =~ s/^\s+|\s+$//g;
    chomp($line);
	      $multi_line_statment .= $line;
	      $line = <INPUT_FH>;
	    }

	  $line =~ s/^\s+//g;
	  $line = $multi_line_statment.$line;
	  $multi_line_statment = "";
	  
	}

    }
    if ( $line =~ /^\s*(module)\s+(\w+)\b/ ) {
			 
			 
             }
		
	elsif ($line =~ /^\s*(endmodule)\b/) {
			 @Module_Wires=();
			 %wire_params=();
			 }
    if (!($line =~m/module/)){
    if ($line =~ m/
	^\s*
	(\S*)  # Cell name
	\s+
	(\S*)  #Instance Name
	\s*
	\(
	(.+)   #ports list
	\)
	\s*
	\;
	\s*
	$
	/x
       ) {
      $cell_name = $1;
      $instance_name = $2;
      @ports = split/,/, $3;
                foreach $a (@ports) {
                    $a =~ /\s*\.([A-Za-z0-9]*?)\(\s*(\S*)\s*\)/;
                    $ports{$1} = $2;
                }
                my $hash_ref = \%ports;
               my $current_object;
my @current_gate_inputs=();
if ( defined( $$hash_ref{"A1"} ) ) {
                        push @current_gate_inputs, $$hash_ref{"A1"};

                    }

  if ( defined( $$hash_ref{"A2"} ) ) {
                        push @current_gate_inputs, $$hash_ref{"A2"};

                    }
                                     if ( defined( $$hash_ref{"B"} ) ) {
                        push @current_gate_inputs, $$hash_ref{"B"};

                    }
  if ( defined( $$hash_ref{"A3"} ) ) {
                        push @current_gate_inputs, $$hash_ref{"A3"};

                    }

  if ( defined( $$hash_ref{"B1"} ) ) {
                        push @current_gate_inputs, $$hash_ref{"B1"};

                    }
                                     if ( defined( $$hash_ref{"B2"} ) ) {
                        push @current_gate_inputs, $$hash_ref{"B2"};

                    }
  if ( defined( $$hash_ref{"A4"} ) ) {
                        push @current_gate_inputs, $$hash_ref{"A4"};

                    }

  if ( defined( $$hash_ref{"E"} ) ) {
                        push @current_gate_inputs, $$hash_ref{"E"};

                    }
                                     if ( defined( $$hash_ref{"I"} ) ) {
                        push @current_gate_inputs, $$hash_ref{"I"};

                    }


  if ( defined( $$hash_ref{"TE"} ) ) {
                        push @current_gate_inputs, $$hash_ref{"TE"};

                    }

  if ( defined( $$hash_ref{"D"} ) ) {
                        push @current_gate_inputs, $$hash_ref{"D"};

                    }
                            
 if ( defined( $$hash_ref{"RN"} ) ) {
                        push @current_gate_inputs, $$hash_ref{"RN"};

                    }
 if ( defined( $$hash_ref{"SN"} ) ) {
                        push @current_gate_inputs, $$hash_ref{"SN"};

                    }
 if ( defined( $$hash_ref{"D"} ) ) {
                        push @current_gate_inputs, $$hash_ref{"D"};

                    }
 if ( defined( $$hash_ref{"CI"} ) ) {
                        push @current_gate_inputs, $$hash_ref{"CI"};

                    }
 if ( defined( $$hash_ref{"A"} ) ) {
                        push @current_gate_inputs, $$hash_ref{"A"};

                    }
                    
                    if ( defined( $$hash_ref{"SI"} ) ) {
                        push @current_gate_inputs, $$hash_ref{"SI"};

                    }
 if ( defined( $$hash_ref{"EN"} ) ) {
                        push @current_gate_inputs, $$hash_ref{"EN"};

                    }
                    

                    if ( defined( $$hash_ref{"I0"} ) ) {
                        push @current_gate_inputs, $$hash_ref{"I0"};

                    }

                    if ( defined( $$hash_ref{"I1"} ) ) {
                        push @current_gate_inputs, $$hash_ref{"I1"};

                    }
                    if ( defined( $$hash_ref{"SE"} ) ) {
                        push @current_gate_inputs, $$hash_ref{"SE"};

                    }
                    if ( defined( $$hash_ref{"I0"} ) &&  defined( $$hash_ref{"I1"} )&& defined( $$hash_ref{"S"} )){
                        push @current_gate_inputs, $$hash_ref{"S"};

                    }

my @current_gate_outputs=();
   
          
                    
if ( defined($$hash_ref{"Q"})  ){

push @current_gate_outputs , $$hash_ref{"Q"};

}

elsif (   defined($$hash_ref{"CO"}) && defined($$hash_ref{"S"})  ){


push @current_gate_outputs,$$hash_ref{"S"};
push @current_gate_outputs, $$hash_ref{"CO"};
}
elsif(defined($$hash_ref{"CO"})){

push @current_gate_outputs , $$hash_ref{"CO"};


}



                    elsif (defined $$hash_ref{"ZN"}){
                    push @current_gate_outputs , $$hash_ref{"ZN"};
                    }
                    elsif (defined $$hash_ref{"Z"}){
                    push @current_gate_outputs , $$hash_ref{"Z"};
                    }
					my $bool_fun=$cell_name;
					if ($cell_name =~ m/\_(\S+)/){$driving=$1;
}
                                        $bool_fun=~s/\_\S+//g;
					$bool_fun=~s/\d+\D*$//g;
					my @updates=();

					my $modified_name="${instance_name}";
						  push @list_of_gates, $modified_name;
push @functions, $bool_fun;                    
$current_object = theCircuit->new(
                        {	
			    processed => $driving,
                            name          => $modified_name,
                            bool_func     => $bool_fun,
                            inputs        => \@current_gate_inputs,
                            outputs        => \@current_gate_outputs,
                            fwdgates => [undef],
                            fwdgates_inst => [undef],
                            count =>$ml_count,
                        }
                    );
					my $indicator=0;
					foreach my $current_gate_output (@current_gate_outputs){
                    if ( exists( $Netlist_Outputs_Hash{$current_gate_output} ) )
                    { 
					my @temp=();
					my @temp_inst=();
					if ($indicator==0){
					
                    push @temp, "PO";
                    
                    push @temp_inst, $current_gate_output;
					}
					else{
					@temp=$current_object->get_fwdgates();
					@temp_inst=$current_object->get_fwdgates_inst();
					push @temp, "PO";
                    
                    push @temp_inst, $current_gate_output;
					}
					$indicator++;
					
					
                      $current_object->set_fwdgates(\@temp);
                      $current_object->set_fwdgates_inst(\@temp_inst);
                    }
					}
                                    $the_circuit{$modified_name} = $current_object;
                $ml_count++;
               
                    
             }       
                

}
      }


        #######end of opening file again

    close INPUT_FH;
    ############################
@functions=uniq(@functions);
print "The functions are @functions\n";
print "Ended with node count $ml_count\n";
foreach my $object ( values %the_circuit ) {  ##### loop through the gates
my $name="";
$name= $object->get_name();

my @current_inputss=$object->get_inputs();


my $limit=0;
$limit=@current_inputss;
my @current_inputs=();
my @current_gate_inputs=();
my @current_gate_inputs_inst=();
my $outer_gate_type=$object->get_bool_func();

for my $i_index (0 .. $#current_inputss)
{
my $in=   $current_inputss[$i_index];
if ( exists( $Netlist_Inputs_Hash{$in} ) )
 {	
 push @current_gate_inputs, "PI";
 push @current_gate_inputs_inst,$in;
 $limit--;
	 
}
else{
	
push @current_inputs, $in;	
}#end if it is not a PI
}# end of looping through the inputs


 if ($limit!=0){ #if my input array is not empty
OUTER: 
foreach my $instance (@list_of_gates)
  {
		   my $current_objectt ="";
		   my @current_outputs=();
	
                   $current_objectt = $the_circuit{$instance};
	  @current_outputs= $current_objectt->get_outputs();
		   my $current_gate_type="";
                   $current_gate_type=$current_objectt->get_bool_func();
		   foreach my $current_output (@current_outputs){
		                      foreach my $input (@current_inputs)
                   {

                   if ($input eq $current_output)
                   {push @current_gate_inputs, $current_gate_type;
                   push @current_gate_inputs_inst, $instance;
                   my @temp=();
                   my @temp_inst=();
                    if ($current_objectt->get_fwdgates()){
                   @temp=$current_objectt->get_fwdgates();
                   @temp_inst=$current_objectt->get_fwdgates_inst();
                   }
                   push @temp, $outer_gate_type;
                   push @temp_inst, $name;
                   @temp = grep defined, @temp;
                   @temp_inst = grep defined, @temp_inst;
                    $current_objectt->set_fwdgates(\@temp);
                    $current_objectt->set_fwdgates_inst(\@temp_inst);
     
                       $the_circuit{ $instance } = $current_objectt;
                   }#the input is a primary output of a gate
                   
                   }
}
}
}#end if my input array is not empty
$object->set_fedbygates(\@current_gate_inputs);

$object->set_fedbygates_inst(\@current_gate_inputs_inst);
  $the_circuit{ $name } = $object;
  
}#end of the outer loop through the gates

foreach my $object ( values %the_circuit ) {  ##### loop through the gates

my @OUts=$object->get_fwdgates();
my @features_array=(0) x 25;
my $driving_id=$object->get_processed();
if ($driving_id eq "X1"){

$features_array[$features_map{"driving"}]=0;
}
elsif ($driving_id eq "X2"){

$features_array[$features_map{"driving"}]=1;

}
$features_array[$features_map{$object->get_bool_func()}]=1;
my @INputs=$object->get_fedbygates();
my $in_degree=@INputs;
$features_array[$features_map{"in_degree"}]=$in_degree;
my %params = map { $_ => 1 } @INputs;
if(exists($params{"PI"})) { 
my $prev=0;
$features_array[$features_map{"PI"}]=($prev+1);


 }
my $name="";
$name= $object->get_name();
my $count=$object->get_count();
my @current_fwd_gates=();
@current_fwd_gates=$object->get_fwdgates_inst();
my $out_degree=@current_fwd_gates;
$features_array[$features_map{"out_degree"}]=$out_degree;
foreach my $elem (@current_fwd_gates){
if (exists ($the_circuit{$elem}))  {
my $current_ob=$the_circuit{$elem};
my $current_count=$current_ob->get_count();
my $inputt=$current_ob->get_bool_func();
my @INNputs=$current_ob->get_fwdgates();
print FH_LINK "$count $current_count\n";
}
}

%params=();
%params = map { $_ => 1 } @OUts;
my $check_flag=0;
if(exists($params{"PO"})) { #$features_array[$features_map{"PO"}]=1;
my $prev=0;
$features_array[$features_map{"PO"}]=($prev+1);
print FH_LINK "$count $count\n";
}

print FH_CELL "$count $name from file $input_file\n";
print FH_COUNT "$count\n";

my $size=@features_array;
print FH_FEAT "@features_array\n";
}#end of the outer loop through the gates
}
close(FH_FEAT);
close(FH_CELL);
close(FH_LINK);
close(FH_COUNT);
my $run_time = time - $start_time;
print STDERR "\nProgram completed in $run_time sec ";

if ($error) {
    print STDERR "with total $error errors.\n\n" and $status = 1;
}
else {
    print STDERR "without error.\n\n";
}

exit $status;

sub PrintWarning {
    warn "WARNING: @_\a\n";
}

sub PrintError {
    ++$error;
    warn "ERROR: @_\a\n";
}

sub PrintFatalError {
    ++$error;
    die "FATAL ERROR: @_\a\n";
}

sub PrintInternalError {
    ++$error;
    die "INTERNAL ERROR: ", (caller)[2], ": @_\a\n";
}
sub uniq {
    my %seen;
    grep !$seen{$_}++, @_;
}
sub PrintDebug {
    my $orig_list_separator = $";
    $" =
      ',';   # To print with separator, @some_list must be outside double-quotes
    warn "DEBUG: ", (caller)[2], ": @_\n" if ($debug);
    $" = $orig_list_separator;    # Dummy " for perl-mode in Emacs
}



__END__

