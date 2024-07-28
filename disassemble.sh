PROFILE=$1
# TARGET=thumbv7em-none-eabihf
TARGET=thumbv8m.main-none-eabihf

OUTPUT_FILE=$PROFILE.$TARGET.txt

if [[ $PROFILE == 'debug' ]]; then
  CARGO_ARGS=""
else
  CARGO_ARGS="--release"
fi

cargo build --target=$TARGET $CARGO_ARGS
arm-none-eabi-objdump -d -S target/$TARGET/$PROFILE/libmicrocheby.rlib | rustfilt > $OUTPUT_FILE
echo Wrote disassembled binary to $OUTPUT_FILE