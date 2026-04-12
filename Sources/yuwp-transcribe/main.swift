import Foundation
import NativeASR

exit(Int32(YuwpTranscribeSupport.runCLI(arguments: Array(CommandLine.arguments.dropFirst()))))
