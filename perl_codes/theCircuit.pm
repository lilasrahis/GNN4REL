package theCircuit;
use strict;
use warnings;
use Data::Dumper;

sub new{
my ($class,$args) = @_;
    my $self = bless { 
			name=> $args->{name},
 			inputs=> $args->{inputs},
 			count=> $args->{count},
 			bool_func=> $args->{bool_func}, # {OR, AND etc}
 			fwdgates=> $args->{fwdgates},
 			fwdgates_inst=> $args->{fwdgates_inst},
 			output => $args->{output},
			processed => $args->{processed},
 			outputs => $args->{outputs},
 			fedbygates=> $args->{fedbygates},
 			fedbygates_inst=> $args->{fedbygates_inst},

                     }, $class;
}

sub get_name{
   my $self = shift;
   return $self->{name};
}
sub set_name{
   my ($self,$new_name) = @_;
   $self->{name} = $new_name;
}



sub get_output{
   my $self = shift;
   return $self->{output};
}
sub set_output{
   my ($self,$new_output) = @_;
   $self->{output} = $new_output;
}

sub get_count{
   my $self = shift;
   return $self->{count};
}
sub set_count{
   my ($self,$new_count) = @_;
   $self->{count} = $new_count;
}
sub get_outputs{
   my $self = shift;
   return @{$self->{outputs}};
}
sub set_outputs{
   my ($self,$new_outputs) = @_;
   $self->{outputs} = $new_outputs;
}

sub get_inputs{
   my $self = shift;
   return @{$self->{inputs}};
}
sub set_inputs{
   my ($self,$new_inputs) = @_;
   $self->{inputs} = $new_inputs;
}

sub get_bool_func{
   my $self = shift;
   return $self->{bool_func};
}
sub set_bool_func{
   my ($self,$new_bool_func) = @_;
   $self->{bool_func} = $new_bool_func;
}

sub get_fwdgates{
   my $self = shift;
   return @{$self->{fwdgates}};
}
sub set_fwdgates{
   my ($self,$new_fwdgates) = @_;
   $self->{fwdgates} = $new_fwdgates;
}
sub get_fwdgates_inst{
   my $self = shift;
   return @{$self->{fwdgates_inst}};
}
sub set_fwdgates_inst{
   my ($self,$new_fwdgates_inst) = @_;
   $self->{fwdgates_inst} = $new_fwdgates_inst;
}

sub get_processed{
   my $self = shift;
   return $self->{processed};
}
sub set_processed{
   my ($self,$new_processed) = @_;
   $self->{processed} = $new_processed;
}

sub get_fedbygates{
   my $self = shift;
   return @{$self->{fedbygates}};
}
sub set_fedbygates{
   my ($self,$new_fedbygates) = @_;
   $self->{fedbygates} = $new_fedbygates;
}
sub get_fedbygates_inst{
   my $self = shift;
   return @{$self->{fedbygates_inst}};
}
sub set_fedbygates_inst{
   my ($self,$new_fedbygates_inst) = @_;
   $self->{fedbygates_inst} = $new_fedbygates_inst;
}


sub to_string{
   my $self = shift;
   return "Name: $self->{name}\nCount: $self->{count}\nInputs: @{$self->{inputs}}\nBool_func: $self->{bool_func}\nfwdgates: @{$self->{fwdgates}}\nProcessed: $self->{processed}\nfedbygates: @{$self->{fedbygates}}\n";
}
1;
