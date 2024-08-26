#!/usr/bin/perl -w

use strict;

my ($language) = @ARGV;

while(<STDIN>) {
    s/\r//g;
    # remove extra spaces
    s/\(/ \(/g;
    s/\)/\) /g; s/ +/ /g;
    s/\) ([\.\!\:\?\;\,])/\)$1/g;
    s/\( /\(/g;
    s/ \)/\)/g;
    s/(\d) \%/$1\%/g;
    s/ :/:/g;
    s/ ;/;/g;
    # normalize unicode punctuation
    s/��/\"/g;
    s/��/\"/g;
    s/��/\"/g;
    s/��/-/g;
    s/��/ - /g; s/ +/ /g;
    s/쨈/\'/g;
    s/([a-z])��([a-z])/$1\'$2/gi;
    s/([a-z])��([a-z])/$1\'$2/gi;
    s/��/\"/g;
    s/��/\"/g;
    s/��/\"/g;
    s/''/\"/g;
    s/쨈쨈/\"/g;
    s/��/.../g;
    # French quotes
    s/혻짬혻/ \"/g;
    s/짬혻/\"/g;
    s/짬/\"/g;
    s/혻쨩혻/\" /g;
    s/혻쨩/\"/g;
    s/쨩/\"/g;
    # handle pseudo-spaces
    s/혻\%/\%/g;
    s/n쨘혻/n쨘 /g;
    s/혻:/:/g;
    s/혻쨘C/ 쨘C/g;
    s/혻cm/ cm/g;
    s/혻\?/\?/g;
    s/혻\!/\!/g;
    s/혻;/;/g;
    s/,혻/, /g; s/ +/ /g;

    # English "quotation," followed by comma, style
    if ($language eq "en") {
	s/\"([,\.]+)/$1\"/g;
    }
    # Czech is confused
    elsif ($language eq "cs" || $language eq "cz") {
    }
    # German/Spanish/French "quotation", followed by comma, style
    else {
	s/,\"/\",/g;	
	s/(\.+)\"(\s*[^<])/\"$1$2/g; # don't fix period at end of sentence
    }

    print STDERR $_ if /癤�/;

    if ($language eq "de" || $language eq "es" || $language eq "cz" || $language eq "cs" || $language eq "fr") {
	s/(\d)혻(\d)/$1,$2/g;
    }
    else {
	s/(\d)혻(\d)/$1.$2/g;
    }
    print $_;
}